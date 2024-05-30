#https://www.mdpi.com/2072-4292/12/14/2260
#https://stats.stackexchange.com/questions/258166/good-accuracy-despite-high-loss-value
#https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch Early stopping
import os
import pickle
import matplotlib.pyplot as plt
import time

from imutils import paths
from files.dataset import SegmentationDataset, EarlyStopper
from files import config

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from tqdm import tqdm

import torch.nn as nn
import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Dice
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import JaccardIndex, MulticlassAccuracy

def binarifyModel(model):
    # Access the weights of the segmentation head
    weights_0 = model.segmentation_head[0].weight[0].data.clone().detach()
    weights_1 = model.segmentation_head[0].weight[1].data.clone().detach()
    weights_2 = model.segmentation_head[0].weight[2].data.clone().detach()
    weights_3 = model.segmentation_head[0].weight[3].data.clone().detach()
    weights_4 = model.segmentation_head[0].weight[4].data.clone().detach()

    bias_0 = model.segmentation_head[0].bias[0].data.clone().detach()
    bias_1 = model.segmentation_head[0].bias[1].data.clone().detach()
    bias_2 = model.segmentation_head[0].bias[2].data.clone().detach()
    bias_3 = model.segmentation_head[0].bias[3].data.clone().detach()
    bias_4 = model.segmentation_head[0].bias[4].data.clone().detach()

    # Add the weights together
    new_weights = torch.add(torch.add(torch.add(weights_0, weights_2), weights_3), weights_4)
    new_bias = torch.add(torch.add(torch.add(bias_0, bias_2), bias_3), bias_4)

    # Create a new convolutional layer
    Conv = nn.Conv2d(16, 2, kernel_size=(1, 1), stride=(1, 1))

    # Set the weights of the convolutional layer
    Conv.weight[0].data = new_weights
    Conv.weight[1].data = weights_1

    Conv.bias[0].data = new_bias
    Conv.bias[1].data = bias_1

    # Replace the segmentation head in the model
    model.segmentation_head[0] = Conv.to('cuda')

    print(model)
    return model

# Function to visualize image, ground truth label, and model prediction
def visualize_sample(image, label, prediction):
    plt.figure(figsize=(12, 4))

    # Denormalize and display the image
    plt.subplot(1, 3, 1)
    plt.imshow(transforms.ToPILImage()(image))
    plt.title('Image')
    plt.axis('off')

    # Display ground truth label
    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap='jet', vmin=0, vmax=4)  # Assuming 5 classes
    plt.title('Ground Truth')
    plt.axis('off')

    # Display model prediction
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='jet', vmin=0, vmax=4)  # Assuming 5 classes
    plt.title('Prediction')
    plt.axis('off')

    plt.show()

def LossFunction(BCE, DICE, weight):
    return(BCE + ( 1 - DICE) * weight)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    H = {"train_loss": [], "test_loss": [], "metrics_train": [], "accuracy_train": [],
          "metrics_validation": [], "accuracy_validation": [], "ConfusionTrain": [], "ConfusionValidation": []}

    file_path = os.path.dirname(os.path.realpath(__file__))
    TrainImagePaths = sorted(list(paths.list_images(os.path.join(file_path,config.train_image))))
    TrainLabelPaths = sorted(list(paths.list_images(os.path.join(file_path, config.train_label))))

    TestImagePaths = sorted(list(paths.list_images(os.path.join(file_path,config.test_image))))
    TestLabelPaths = sorted(list(paths.list_images(os.path.join(file_path, config.test_label))))

    metric_train = JaccardIndex(task="multiclass", num_classes=2, average=None).to(config.device)
    metric_validation = JaccardIndex(task="multiclass", num_classes=2, average=None).to(config.device)

    AccuracyMetric_train = MulticlassAccuracy(num_classes=2, average=None).to(config.device)
    AccuracyMetric_validation = MulticlassAccuracy(num_classes=2, average=None).to(config.device)

    ConfusionMatrixTrain = MulticlassConfusionMatrix(num_classes=2, normalize="true").to(config.device)
    ConfusionMatrixValidation = MulticlassConfusionMatrix(num_classes=2, normalize="true").to(config.device)
    
    #preprocess_input = get_preprocessing_fn('mobilenet_v2', pretrained='imagenet')

    trainSteps = len(TrainImagePaths) // config.batch_size
    testSteps = len(TestImagePaths) // config.batch_size
    
    totalPixels = torch.FloatTensor([797*10e6+0.3*10e6+45.7*10e6, 9.1*10e6+50.4*10e6])
    weights = 1 - (totalPixels / totalPixels.sum())
    #weights = torch.FloatTensor([1,6])
    #model = smp.DeepLabV3Plus('mobilenet_v2', classes=2, activation='identity', encoder_depth=5, encoder_weights = "imagenet").to(config.device)
    #BCELossFunc = BCEWithLogitsLoss(pos_weight=torch.FloatTensor([6])).to(config.device)
    LossFunc = CrossEntropyLoss().to(config.device)
    #DiceLoss = Dice().to(config.device)
    early_stopper = EarlyStopper(patience=10, min_delta=0, lossvalue=None)
    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()
    model = torch.load(os.path.join(file_path,config.model)).to(config.device)
    model = binarifyModel(model)
    print(model)
    lr  = config.init_LR
    opt = Adam(model.parameters(), lr=lr)
    
    for param in model.encoder.parameters(): #Freeze weights
        param.requires_grad = False
    #######################################

    print(f"[INFO] found {len(TrainImagePaths)} examples in the training set...")
    print(f"[INFO] found {len(TestImagePaths)} examples in the test set...")
    print("[INFO] training the network...")

    startTime = time.time()
    iteration = 1
    
    for e in tqdm(range(config.num_epochs)):
        totalTrainLoss = 0 
        totalTestLoss = 0
        
        trainDS = SegmentationDataset(imagePaths=TrainImagePaths, labelPaths=TrainLabelPaths, T = True)

        trainLoader = DataLoader(trainDS, shuffle=True,
                            batch_size = config.batch_size, pin_memory = config.pin_memory,
                            num_workers = os.cpu_count())
        
        model.train()

        for(i, (x,y)) in enumerate(trainLoader):
            x, y = x.to(config.device), y.to(config.device)
            x = x.type(torch.FloatTensor).to(config.device)
            #print(x.size())
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(x)
                #loss = LossFunction(LossFunc(pred, y), DiceLoss(pred,y), weight=weights[0].to(torch.long))
                loss = LossFunc(pred, y)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            totalTrainLoss += loss

            metric_train.update(torch.argmax(pred, dim=1), y)
            AccuracyMetric_train.update(torch.argmax(pred, dim=1), y)
            ConfusionMatrixTrain.update(torch.argmax(pred, dim=1), y)
            #visualize_sample(x[0].cpu().numpy().transpose(1, 2, 0), torch.argmax(y, dim=1)[0].cpu().numpy(), torch.argmax(pred[0], dim=0).cpu().numpy())
        with torch.no_grad():
            model.eval()

            testDS = SegmentationDataset(imagePaths=TestImagePaths, labelPaths=TestLabelPaths, T = True)

            testLoader = DataLoader(testDS, shuffle=False,
                            batch_size=config.batch_size, pin_memory=config.pin_memory,
                            num_workers=os.cpu_count())
            
            for (x, y) in testLoader:
                x, y = x.to(config.device), y.to(config.device)
                x = x.type(torch.FloatTensor).to(config.device)
                pred = model(x)

                #loss = LossFunc(pred, y)
                totalTestLoss += loss
                #totalTestLoss += LossFunction(LossFunc(pred, y), DiceLoss(pred,y), weights[0].to(torch.long))
                metric_validation.update(torch.argmax(pred, dim=1), y)
                AccuracyMetric_validation.update(torch.argmax(pred, dim=1), y)
                ConfusionMatrixValidation.update(torch.argmax(pred, dim=1), y)
        
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        print("[INFO EPOCH: {}/{}]".format(e +1, config.num_epochs))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

        if early_stopper.early_stop(avgTrainLoss):             
            
            lr = lr*0.1
            if lr < 5e-6:
                H["metrics_validation"].append(metric_validation.compute())
                H["accuracy_validation"].append(AccuracyMetric_validation.compute())
                H["metrics_train"].append(metric_train.compute())
                H["accuracy_train"].append(AccuracyMetric_train.compute())
                H["ConfusionTrain"].append(ConfusionMatrixTrain.compute())
                H["ConfusionValidation"].append(ConfusionMatrixValidation.compute())

                with open(os.path.join(config.output, 'H.pkl'), 'wb') as fp:
                    pickle.dump(H, fp)

                torch.save(model, os.path.join(file_path, config.model))
                print(f'Stopping model at {iteration} epochs')
                print('dictionary saved successfully to file')
                print('model saved')
                break
            else:
                opt = Adam(model.parameters(), lr=lr)
                print(f'Early stopping at {iteration} epochs, lowering learning rate to {lr}')
                early_stopper.counter = 0

        
        if iteration % 5 == 0:
            H["metrics_validation"].append(metric_validation.compute())
            H["accuracy_validation"].append(AccuracyMetric_validation.compute())
            H["metrics_train"].append(metric_train.compute())
            H["accuracy_train"].append(AccuracyMetric_train.compute())
            H["ConfusionTrain"].append(ConfusionMatrixTrain.compute())
            H["ConfusionValidation"].append(ConfusionMatrixValidation.compute())
            
            print("[INFO] Model Metrics are as shown: \n")
            print("Metrictrain: ", metric_train.compute(), " Accuracytrain: ", AccuracyMetric_train.compute())
            print("Metricvalidation: ", metric_validation.compute(), " Accuracyvalidation: ", AccuracyMetric_validation.compute())
            
            with open(os.path.join(config.output, 'H.pkl'), 'wb') as fp:
                pickle.dump(H, fp)
            
            torch.save(model, os.path.join(file_path, config.model))
            print('dictionary saved successfully to file')
            print('model saved')

        metric_train.reset()
        AccuracyMetric_train.reset()
        metric_validation.reset()
        AccuracyMetric_validation.reset()
        ConfusionMatrixValidation.reset()
        ConfusionMatrixTrain.reset()
        iteration +=1
        
    
    endTime = time.time()

    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

    with open(os.path.join(config.output, 'H.pkl'), 'wb') as fp:
        pickle.dump(H, fp)
    print('dictionary saved successfully to file')
    
    # Plotting
    plt.style.use("ggplot")
    plt.figure()

    # Plotting loss
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="validation_loss")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(file_path, config.plot_path))
 
    # Plotting metrics and accuracy
    plt.figure()
    fig1, ax1 = metric_validation.plot(H["metrics_validation"])
    fig2, ax2 = AccuracyMetric_validation.plot(H["accuracy_validation"])
    fig3, ax3 = metric_train.plot(H["metrics_train"])
    fig4, ax4 = AccuracyMetric_train.plot(H["accuracy_train"])
    fig5, ax5 = ConfusionMatrixTrain.plot(H["ConfusionTrain"][-1])
    fig6, ax6 = ConfusionMatrixValidation.plot(H["ConfusionValidation"][-1])
   
    fig1.savefig(os.path.join(file_path, config.validationMetric_path))
    fig2.savefig(os.path.join(file_path, config.validationAccuracy_path))
    fig3.savefig(os.path.join(file_path, config.trainMetric_path))
    fig4.savefig(os.path.join(file_path, config.trainAccuracy_path))
    fig5.savefig(os.path.join(file_path, config.trainConf_path))
    fig6.savefig(os.path.join(file_path, config.validationConf_path))
    torch.save(model, os.path.join(file_path, config.model))