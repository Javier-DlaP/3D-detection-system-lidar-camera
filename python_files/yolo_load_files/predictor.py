import os
import time
from torchvision.models import vgg

from func import *
from eval import *

def Yolo5_det(imgs,model,threshold_conf=0.6):

    # Images
    model.classes = [0,1,2] #Person(0),bicycle(1),car(2),motorcycle(3),bus(5),truck(7),traffic light(9),stop(11)
    model.conf = threshold_conf
    #Adding classes = n in torch.hub.load will change the output layers (must retrain with the new number of classes)

    t_ini = time.time()

    # Inference
    results = model(imgs)
    #print(results.pred)
    #print('\n')

    t_end = time.time()
    elapsed = (t_end - t_ini)/len(imgs)
    print("Yolo_time (avg sec per image): {}".format(elapsed))

    return results

def load_model_est(dir):

    model_lst = [x for x in sorted(os.listdir(dir)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file
        model = Model(features=my_vgg.features, bins=2).cuda()
        #checkpoint = torch.load(dir + '/%s'%model_lst[-1])
        #model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
def get_estimations(KITTI_DATASET, WORK_PATH, WEIGHTS_PATH, batch_size=400, threshold=0.6):
    
    def estimation_batch():
        
        for i in range(len(imgs)):

            estimations_img = []

            elem_box = detections.pred[i]

            for element in elem_box:

                name = detections.names[int(element.data[5])]
                yolo_conf = element[4]

                box_2d = [(int(element[0].item()),int(element[1].item())),(int(element[2].item()),int(element[3].item())), name, yolo_conf.cpu().numpy().tolist()]
                estimations_img.append(box_2d)

            estimations.append(estimations_img)

    dir1 = KITTI_DATASET+'images/'
    dir2 = KITTI_DATASET+'calib/'

    names = sorted(os.listdir(dir1))
    n_images = len(os.listdir(dir1))
    par = sorted(os.listdir(dir2))
    
    n_batches = n_images//batch_size
    n_remaining = n_images - n_batches*batch_size 
    
    torch.hub.set_dir(WORK_PATH)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    model = load_model_est(WEIGHTS_PATH)

    estimations = []
    
    for batch in range(n_batches):
        
        print('Batch: '+str(batch) + ' ('+str((batch*batch_size)*100/n_images)+'%)')
        
        imgs = [dir1 + name for name in names[batch*batch_size:batch*batch_size+batch_size]]     # batch of images 
        detections = Yolo5_det(imgs,yolo_model,threshold_conf=threshold)
        
        estimation_batch()
            
    print('Last batch')
    imgs = [dir1 + name for name in names[n_batches*batch_size:n_images]]     # batch of images 
    detections = Yolo5_det(imgs,yolo_model,threshold_conf=threshold)
    
    estimation_batch()
        
    return estimations