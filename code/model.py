from __future__ import division
import os
import time
from glob import glob
from ops import *
from utils import *
import numpy as np
import copy
from skimage.transform import resize

class counting_model(object):
    def __init__(self, sess, param_set):
        self.sess           = sess
        self.phase          = param_set['phase']
        self.batch_size     = param_set['batch_size']
        self.inputI_width_size    = param_set['inputI_width_size']
        self.inputI_height_size    = param_set['inputI_height_size']
        self.inputI_chn     = param_set['inputI_chn']
        self.output_chn     = param_set['output_chn']
        self.trainImagePath  = param_set['trainImagePath']
        self.trainDmapPath  = param_set['trainDmapPath']
        self.trainPmapPath  = param_set['trainPmapPath']
        self.testImagePath  = param_set['testImagePath']
        self.testDmapPath  = param_set['testDmapPath']
        self.testPmapPath  = param_set['testPmapPath']
        self.chkpoint_dir   = param_set['chkpoint_dir']
        self.lr             = param_set['learning_rate']
        self.beta1          = param_set['beta1']
        self.epoch          = param_set['epoch']
        self.model_name     = param_set['model_name']
        self.save_intval    = param_set['save_intval']
        self.labeling_dir   = param_set['labeling_dir']
        self.result_dir   = param_set['result_dir']
        self.log_dir   = param_set['log_dir']
        self.density_level  = param_set['density_level']
        self.inputI_size = [self.inputI_width_size,self.inputI_height_size]
        # build model graph
        self.build_model()
        

    def focal_loss_num_func(self, logits, labels, alpha=0.25, gamma=2.0):
        """
        Loss = weighted * -target*log(softmax(logits))
        :param logits: probability score
        :param labels: ground_truth
        :return: softmax-weighted loss
        """
        gt = tf.one_hot(labels,self.density_level)
        softmaxpred = tf.nn.softmax(logits)
        loss = 0
        for i in range(self.density_level):
            gti = gt[:,i]
            predi = softmaxpred[:,i]
            loss = loss+ tf.reduce_mean(gti*tf.pow(1 - predi, gamma)* tf.log(tf.clip_by_value(predi, 0.005, 1)))
        return -loss/self.density_level

    
    def focal_loss_func(self, logits, labels, alpha=0.25, gamma=2.0):
        """
        Loss = weighted * -target*log(softmax(logits))
        :param logits: probability score
        :param labels: ground_truth
        :return: softmax-weighted loss
        """
        gt = tf.one_hot(labels,2)
        softmaxpred = tf.nn.softmax(logits)
        loss = 0
        for i in range(2):
            gti = gt[:,:,:,i]
            predi = softmaxpred[:,:,:,i]
            weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(gt))
            loss = loss+ tf.reduce_mean(weighted *gti* tf.pow(1 - predi, gamma)* tf.log(tf.clip_by_value(predi, 0.005, 1)))
        return -loss/2

    def l1_loss(self, prediction, ground_truth, weight_map=None):
        """
        :param prediction: the current prediction of the ground truth.
        :param ground_truth: the measurement you are approximating with regression.
        :return: mean of the l1 loss.
        """
        absolute_residuals = tf.abs(tf.subtract(prediction[:,:,:,0], ground_truth))
        if weight_map is not None:
            absolute_residuals = tf.multiply(absolute_residuals, weight_map)
            sum_residuals = tf.reduce_sum(absolute_residuals)
            sum_weights = tf.reduce_sum(weight_map)
        else:
            sum_residuals = tf.reduce_sum(absolute_residuals)
            sum_weights = tf.size(absolute_residuals)
        return tf.truediv(tf.cast(sum_residuals, dtype=tf.float32),
                          tf.cast(sum_weights, dtype=tf.float32))
    
    
    def l2_loss(self, prediction, ground_truth):
        """
        :param prediction: the current prediction of the ground truth.
        :param ground_truth: the measurement you are approximating with regression.
        :return: sum(differences squared) / 2 - Note, no square root
        """
    
        residuals = tf.subtract(prediction[:,:,:,0], ground_truth)
        sum_residuals = tf.nn.l2_loss(residuals)
        sum_weights = tf.size(residuals)
        return tf.truediv(tf.cast(sum_residuals, dtype=tf.float32),
                          tf.cast(sum_weights, dtype=tf.float32))
 

    def seg_dice(self, move_img, refer_img):
        # list of classes
        c_list = np.unique(refer_img)

        dice_c = []
        for c in range(len(c_list)):
            # intersection
            ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
            # sum
            sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
            dice_c.append((2.0 * ints) / sums)

        return dice_c

    # build model graph
    def build_model(self):
        # input
        self.input_Img = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.inputI_chn], name='input_Img')
        self.input_Dmap = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='input_Dmap')
        self.input_Pmap = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='input_Pmap')
        self.input_num = tf.placeholder(dtype=tf.int32, shape=[None], name='input_num')

        print('Model:' + self.model_name)
        self.pred_pprob, self.soft_pprob, self.pred_plabel, self.pred_dprob, self.pred_num = self.focus_network(self.input_Img)

        # =========density estimation loss=========
        self.density_loss = 10*self.l1_loss(self.pred_dprob, self.input_Dmap)+self.l2_loss(self.pred_dprob, self.input_Dmap)
        # =========segmentation loss=========
        self.segment_loss = 10*self.focal_loss_func(self.pred_pprob, self.input_Pmap)
        # =========global density prediction loss=========
        self.global_density_loss = self.focal_loss_num_func(self.pred_num,self.input_num)
        # =========density estimation loss=========
        self.total_loss = self.density_loss + self.segment_loss + self.global_density_loss
        
        #tf.summary.scalar('training_loss',self.total_loss)
        
        # trainable variables
        self.u_vars = tf.trainable_variables()

        # create model saver
        self.saver = tf.train.Saver(max_to_keep=1000)
    
    def focus_network(self, inputI):
        phase_flag = True
        concat_dim = 3
    	 #***************encoder level0***************        
        conv1 = conv_bn_relu(input=inputI, output_chn=16, kernel_size=7, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='conv1')
        print("output of encoder level0:")
        print(conv1.get_shape())

    	 #***************encoder level1***************        
        res_block1 = bottleneck_block(input=conv1, input_chn=16, output_chn=16, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block1')
        print("output of encoder level1:")
        print(res_block1.get_shape())
        
        #*************encoder level2*************** 
        res_block2 = bottleneck_block(input=res_block1, input_chn=16, output_chn=32, kernel_size=3, stride=2, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block2')
        print("output of encoder level2:")
        print(res_block2.get_shape())
        
        #*************encoder level3*************** 
        res_block3 = bottleneck_block(input=res_block2, input_chn=32, output_chn=64, kernel_size=3, stride=2, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block3')
        res_block3 = bottleneck_block(input=res_block3, input_chn=64, output_chn=64, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block3_1')
        print("output of encoder level3:")
        print(res_block3.get_shape())
        
        #*************encoder level4*************** 
        res_block4 = bottleneck_block(input=res_block3, input_chn=64, output_chn=96, kernel_size=3, stride=2, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block4')
        res_block4 = bottleneck_block(input=res_block4, input_chn=96, output_chn=96, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block4_1')
        print("output of encoder level4:")
        print(res_block4.get_shape())
        
        #*************encoder level5*************** 
        res_block5 = bottleneck_block(input=res_block4, input_chn=96, output_chn=96, kernel_size=3, stride=1, dilation=(2,2), use_bias=False, is_training=phase_flag, name='res_block5') 
        res_block5 = bottleneck_block(input=res_block5, input_chn=96, output_chn=96, kernel_size=3, stride=1, dilation=(2,2), use_bias=False, is_training=phase_flag, name='res_block5_1') 
        print("output of encoder level5:")
        print(res_block5.get_shape())
        
        #*************encoder level6*************** 
        res_block6 = bottleneck_block(input=res_block5, input_chn=96, output_chn=96, kernel_size=3, stride=1, dilation=(4,4), use_bias=False, is_training=phase_flag, name='res_block6') 
        res_block6 = bottleneck_block(input=res_block6, input_chn=96, output_chn=96, kernel_size=3, stride=1, dilation=(4,4), use_bias=False, is_training=phase_flag, name='res_block6_1')
        print("output of encoder level6:")
        print(res_block6.get_shape())        
        
        #*************decoder level7***************
        res_block7 = bottleneck_block(input=res_block6, input_chn=96, output_chn=96, kernel_size=3, stride=1, dilation=(2,2), use_bias=False, is_training=phase_flag, name='res_block7')
        res_block7 = bottleneck_block(input=res_block7, input_chn=96, output_chn=96, kernel_size=3, stride=1, dilation=(2,2), use_bias=False, is_training=phase_flag, name='res_block7_1')
        print("output of decoder level7:")
        print(res_block7.get_shape())
        
        #*************decoder level8***************
        res_block8 = bottleneck_block(input=res_block7, input_chn=96, output_chn=96, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block8')
        res_block8 = bottleneck_block(input=res_block8, input_chn=96, output_chn=96, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block8_1')
        print("output of decoder level8:")
        print(res_block8.get_shape())
        
        #*************decoder level9***************
        concat2 = tf.concat([res_block4, res_block5, res_block7, res_block8], axis=concat_dim, name='concat2')
        res_block9 = conv_bn_relu_x2(input=concat2, output_chn=96, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block9')
        res_block9 = conv_bn_relu_x2(input=res_block9, output_chn=96, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='res_block9_1')
        print("output of decoder level9:")
        print(res_block9.get_shape())
        
        #*************decoder level10***************     
        deconv1_upsample = deconv_bn_relu(input=res_block9, output_chn=64, kernel_size=4, stride=2, is_training=phase_flag, name='deconv1_upsample')
        deconv1_conv1 = conv_bn_relu_x2(input=deconv1_upsample, output_chn=64, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='deconv1_conv1')
        print("output of decoder level10:")
        print(deconv1_conv1.get_shape())
        
        #*************decoder level11***************
        deconv2_upsample = deconv_bn_relu(input=deconv1_conv1, output_chn=32, kernel_size=4, stride=2, is_training=phase_flag, name='deconv2_upsample')
        deconv2_conv1 = conv_bn_relu_x2(input=deconv2_upsample, output_chn=32, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='deconv2_conv1')
        print("output of decoder level11:")
        print(deconv2_conv1.get_shape())
        
        #*************decoder level12***************
        deconv3_upsample = deconv_bn_relu(input=deconv2_conv1, output_chn=16, kernel_size=4, stride=2, is_training=phase_flag, name='deconv3_upsample')
        deconv3_conv1 = conv_bn_relu(input=deconv3_upsample, output_chn=16, kernel_size=3, stride=1, dilation=(1,1), use_bias=False, is_training=phase_flag, name='deconv3_conv1')
        print("output of decoder level12:")
        print(deconv3_conv1.get_shape())
		
        pred_pprob = conv2d(input=deconv3_conv1, output_chn=2, kernel_size=1, stride=1, dilation=(1,1), use_bias=True, name='pred_pprob')                   
        soft_pprob = tf.nn.softmax(pred_pprob, name='soft_pred_pprob')
        pred_plabel = tf.argmax(soft_pprob, axis=3, name='argmax')

        pre_bilinear = bilinear_pooling(deconv3_conv1)
        pred_number = tf.layers.dense(pre_bilinear, self.density_level,name='pred_number')
        
        num_attention = tf.layers.dense(pre_bilinear, 16,name='num_attention')
        soft_num_attention = tf.sigmoid(num_attention,name='soft_num_attention')

        pred_dprob_mul = couple_map(tf.stop_gradient(soft_pprob), deconv3_conv1, soft_num_attention)
        pred_dprob = conv2d(input=pred_dprob_mul, output_chn=1, kernel_size=1, stride=1, dilation=(1,1), use_bias=True, name='pred_dprob')        
                          
        return pred_pprob, soft_pprob, pred_plabel, pred_dprob, pred_number

    # train function
    def train(self):
        
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss, var_list=self.u_vars)
        
        # initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        #self.merged = tf.summary.merge_all()
        
        # save .log
        #self.log_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        log_file = open(self.result_dir+'/'+self.model_name+"_log.txt", "w")

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS\n")
            log_file.write(" [*] Load SUCCESS\n")
        else:
            print(" [!] Load failed...\n")
            log_file.write(" [!] Load failed...\n")

        img_list = glob('{}/*.jpg'.format(self.trainImagePath))
        img_list.sort()
        dmap_list = glob('{}/*.mat'.format(self.trainDmapPath))
        dmap_list.sort()        
        pmap_list = glob('{}/*.mat'.format(self.trainPmapPath))
        pmap_list.sort() 
        
        self.test_training(0, log_file)
        
        rand_idx = np.arange(len(img_list))
        start_time = time.time()
        for epoch in np.arange(self.epoch):
            np.random.shuffle(rand_idx)
            epoch_total_loss = 0.0
            for i_dx in rand_idx:
                # train batch
                img_path = img_list[i_dx]
                dmap_path = dmap_list[i_dx]
                pmap_path = pmap_list[i_dx]
                batch_img, batch_dmap, batch_pmap, batch_num = get_batch_patches(img_path, dmap_path, pmap_path, self.inputI_size, self.batch_size) 
                
                _, cur_train_loss = self.sess.run([u_optimizer, self.total_loss], feed_dict={self.input_Img: batch_img,self.input_Dmap: batch_dmap,self.input_Pmap: batch_pmap,self.input_num: batch_num})
                
                #count += 1
                #self.log_writer.add_summary(summary, count)
                
                epoch_total_loss += cur_train_loss
            
            #if np.mod(epoch+1, 2) == 0:
            print("Epoch: [%d] time: %4.4f, train_loss: %.8f\n" % (epoch+1, time.time() - start_time, epoch_total_loss/len(img_list)))
            log_file.write("Epoch: [%d] time: %4.4f, train_loss: %.8f\n" % (epoch+1, time.time() - start_time, epoch_total_loss/len(img_list)))
            log_file.flush()
            self.test_training(epoch+1, log_file)
            start_time = time.time()

            if epoch+1 > 0:#np.mod(epoch+1, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, epoch+1)

        log_file.close()
        
    def test_training(self, step, log_file):
        
        test_img_list = glob('{}/*.jpg'.format(self.testImagePath))
        test_img_list.sort()        
        test_dmap_list = glob('{}/*.mat'.format(self.testDmapPath))
        test_dmap_list.sort()
        test_pmap_list = glob('{}/*.mat'.format(self.testPmapPath))
        test_pmap_list.sort()
        
        all_mae = np.zeros([len(test_img_list)])
        all_rmse = np.zeros([len(test_img_list)])  
        all_dice = np.zeros([len(test_img_list), 2])
        
        for k in range(0, len(test_img_list)): 
            #print k
            img_path = test_img_list[k]
            dmap_path = test_dmap_list[k]
            pmap_path = test_pmap_list[k]                    
            img_data, dmap_data, pmap_data = load_data_pairs(img_path, dmap_path, pmap_path)
            
            w,h,c = img_data.shape
            w = min(int(w/8)*8,1920)
            h = min(int(h/8)*8,1920)
            
            img_data = resize(img_data, (w,h,c),preserve_range=True)
            img_data = img_data.reshape(1,w,h,c)
            dmap_data = dmap_data/100.0
            pmap_data = resize(pmap_data, (w,h),preserve_range=True)
            pmap_data[pmap_data<1] = 0
            pmap_data = pmap_data.reshape(1,w,h)
            
            predicted_label, soft_pprob, pred_plabel = self.sess.run([self.pred_dprob,self.soft_pprob, self.pred_plabel], feed_dict={self.input_Img: img_data})
            predicted_label /= 100.0
                        
            k_dice_c = self.seg_dice(pred_plabel, pmap_data)
            all_dice[k, :] = np.asarray(k_dice_c)
            all_mae[k] = abs(np.sum(predicted_label)-np.sum(dmap_data))
            all_rmse[k] = pow((np.sum(predicted_label)-np.sum(dmap_data)),2)

        mean_dice = np.mean(all_dice, axis=0)        
        mean_mae = np.mean(all_mae, axis=0)
        mean_rmse = pow(np.mean(all_rmse, axis=0),0.5)
        print("Epoch: [%d], mae: %s, rmse:%s, dice:%s\n"%(step,mean_mae,mean_rmse,mean_dice))
        log_file.write("Epoch: [%d], mae: %s, rmse:%s, dice:%s\n"%(step,mean_mae,mean_rmse,mean_dice))
        log_file.flush()            
    
    def test(self):
        
        print("Starting test Process:\n")
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")  

        # get file list of testing dataset
        test_img_list = glob('{}/*.jpg'.format(self.testImagePath))
        test_img_list.sort()        
        test_dmap_list = glob('{}/*.mat'.format(self.testDmapPath))
        test_dmap_list.sort()
        test_pmap_list = glob('{}/*.mat'.format(self.testPmapPath))
        test_pmap_list.sort()
        
        all_mae = np.zeros([len(test_img_list)])
        all_rmse = np.zeros([len(test_img_list)])  
        all_dice = np.zeros([len(test_img_list), 2])
        
        save_labeling_dir = self.labeling_dir+'/'+self.model_name
        if not os.path.exists(save_labeling_dir):
            os.makedirs(save_labeling_dir+'/dmap')
            os.makedirs(save_labeling_dir+'/pmap')
        
        for k in range(0, len(test_img_list)): 
            #print k
            img_path = test_img_list[k]
            dmap_path = test_dmap_list[k]
            pmap_path = test_pmap_list[k]                    
            img_data, dmap_data, pmap_data = load_data_pairs(img_path, dmap_path, pmap_path)
            
            name_index=test_img_list[k].rfind('/')
            name_index_1=test_img_list[k].rfind('.')
            file_name=test_img_list[k][name_index+1:name_index_1]
            
            w,h,c = img_data.shape
            w = min(int(w/8)*8,1920)
            h = min(int(h/8)*8,1920)
            
            img_data = resize(img_data, (w,h,c),preserve_range=True)
            img_data = img_data.reshape(1,w,h,c)
            dmap_data = dmap_data/100.0
            pmap_data = resize(pmap_data, (w,h),preserve_range=True)
            pmap_data[pmap_data<1] = 0
            pmap_data = pmap_data.reshape(1,w,h)
            
            predicted_label, soft_pprob, pred_plabel = self.sess.run([self.pred_dprob,self.soft_pprob, self.pred_plabel], feed_dict={self.input_Img: img_data})
            predicted_label /= 100.0
               
            labeling_path = os.path.join(save_labeling_dir+'/dmap', ('DMAP_' + file_name))
            SaveDmap(predicted_label[0,:,:,0], labeling_path)
                
            labeling_path = os.path.join(save_labeling_dir+'/pmap', ('PMAP_' + file_name))
            SavePmap(soft_pprob[0,:,:,1], labeling_path)
                        
            k_dice_c = self.seg_dice(pred_plabel, pmap_data)
            all_dice[k, :] = np.asarray(k_dice_c)
            all_mae[k] = abs(np.sum(predicted_label)-np.sum(dmap_data))
            all_rmse[k] = pow((np.sum(predicted_label)-np.sum(dmap_data)),2)

        mean_dice = np.mean(all_dice, axis=0)        
        mean_mae = np.mean(all_mae, axis=0)
        mean_rmse = pow(np.mean(all_rmse, axis=0),0.5)
        print("mae: %s, rmse:%s, dice:%s\n"%(mean_mae,mean_rmse,mean_dice))

    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
