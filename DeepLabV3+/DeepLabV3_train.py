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
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import JaccardIndex, MulticlassAccuracy
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from sklearn.utils import class_weight
import numpy as np

# Function to visualize image, ground truth label, and model prediction

def visualize_sample(image, label, prediction):
    plt.figure(figsize=(12, 4))

    # Denormalize and display the image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def DynamicLossWeights(labels):
    classes, class_count = np.unique(labels, return_counts=True)
    total_pixels = labels.size
    class_weights = 1 - (class_count / total_pixels)
    class_weights = class_weights.sum() / class_weights
    return torch.tensor(class_weights, dtype=torch.float16)

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    H = {"train_loss": [], "test_loss": [], "metrics_train": [], "accuracy_train": [],
          "metrics_validation": [], "accuracy_validation": [], "ConfusionTrain": [], "ConfusionValidation": []}
    
    file_path = os.path.dirname(os.path.realpath(__file__))
    TrainImagePaths = sorted(list(paths.list_images(os.path.join(file_path,config.train_image))))
    TrainLabelPaths = sorted(list(paths.list_images(os.path.join(file_path, config.train_label))))

    TestImagePaths = sorted(list(paths.list_images(os.path.join(file_path,config.test_image))))
    TestLabelPaths = sorted(list(paths.list_images(os.path.join(file_path, config.test_label))))

    metric_train = JaccardIndex(task="multiclass", num_classes=5, average=None).to(config.device)
    metric_validation = JaccardIndex(task="multiclass", num_classes=5, average=None).to(config.device)

    AccuracyMetric_train = MulticlassAccuracy(num_classes=5, average=None).to(config.device)
    AccuracyMetric_validation = MulticlassAccuracy(num_classes=5, average=None).to(config.device)

    ConfusionMatrixTrain = MulticlassConfusionMatrix(num_classes=5, normalize="true").to(config.device)
    ConfusionMatrixValidation = MulticlassConfusionMatrix(num_classes=5, normalize="true").to(config.device)
    
    trainDS = SegmentationDataset(imagePaths=TrainImagePaths, labelPaths=TrainLabelPaths, T = True)
    testDS = SegmentationDataset(imagePaths=TestImagePaths, labelPaths=TestLabelPaths, T = True)

    trainSteps = len(trainDS) // config.batch_size
    testSteps = len(testDS) // config.batch_size
    
    totalPixels = torch.FloatTensor([797.7*10e6, 9.1*10e6, 50.4*10e6, 0.3*10e6, 45.7*10e6])

    #weights_normalized = torch.tensor([10e-4, 10e-3, 5e-3, 10e-3, 2e-3])
    weights = 1 - ( totalPixels / totalPixels.sum())

    #preprocess_input = get_preprocessing_fn('mobilenet_v2', pretrained='imagenet')

    model = smp.DeepLabV3Plus('mobilenet_v2', classes=5, activation=None, encoder_depth=5, encoder_weights = "imagenet").to(config.device)
    lossFunc = CrossEntropyLoss().to(config.device)
    early_stopper = EarlyStopper(patience=25, min_delta=0, lossvalue=None)

    #model = torch.load(os.path.join(file_path, config.model)).to(config.device)
    lr = config.init_LR
    print(lr)
    opt = Adam(model.parameters(), lr=lr)
    
    for param in model.encoder.parameters(): #Freeze weights
        param.requires_grad = True
    #######################################

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")
    print("[INFO] training the network...")

    startTime = time.time()
    iteration = 1
    
    for e in tqdm(range(config.num_epochs)):
        totalTrainLoss = 0 
        totalTestLoss = 0
        
        testDS = SegmentationDataset(imagePaths = TestImagePaths, labelPaths = TestLabelPaths, T = True)
        trainDS = SegmentationDataset(imagePaths = TrainImagePaths, labelPaths = TrainLabelPaths, T = True)
        
        trainLoader = DataLoader(trainDS, shuffle=True,
                            batch_size = config.batch_size, pin_memory = config.pin_memory,
                            num_workers = os.cpu_count())

        model.train()

        for(i, (x,y)) in enumerate(trainLoader):
            x, y = x.to(config.device), y.to(config.device)
            x = x.type(torch.FloatTensor).to(config.device)
            #print(x.min(), x.max())
            #x = (x * 255).type(torch.LongTensor)
            #x = preprocess_input(x.cpu().numpy().transpose(0,2,3,1))
            #x = torch.from_numpy(x.transpose(0,3,1,2)).type(torch.FloatTensor).to(config.device)

            #weights = DynamicLossWeights(y.cpu().numpy()).to(config.device)
            pred = model(x)
            
            #loss = torch.mean(lossFunc(pred, y)*weights)
            loss = lossFunc(pred,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss
            metric_train.update(torch.argmax(pred, dim=1), y)
            AccuracyMetric_train.update(torch.argmax(pred, dim=1), y)
            ConfusionMatrixTrain.update(torch.argmax(pred, dim=1), y)
 
            #visualize_sample(x[0].cpu().numpy().transpose(1, 2, 0), y[0].detach().cpu().numpy(), torch.argmax(pred[0], dim=0).detach().cpu().numpy())

        testLoader = DataLoader(testDS, shuffle=False,
                            batch_size=config.batch_size, pin_memory=config.pin_memory,
                            num_workers=os.cpu_count())
        
        with torch.no_grad():
            model.eval()
            
            for (x, y) in testLoader:
                (x, y) = (x.to(config.device), y.to(config.device))
                x = x.type(torch.FloatTensor).to(config.device)
                #x = (x * 255).type(torch.LongTensor)
                #x = preprocess_input(x.cpu().numpy().transpose(0,2,3,1))
                #x = torch.from_numpy(x.transpose(0,3,1,2)).type(torch.FloatTensor).to(config.device)
                weights = DynamicLossWeights(y.cpu().numpy()).to(config.device)
                pred = model(x)
                #loss = torch.mean(lossFunc(pred, y)*weights)
                loss = lossFunc(pred,y)

                totalTestLoss += loss
                #visualize_sample(x[0].cpu().numpy().transpose(1, 2, 0), y[0].detach().cpu().numpy(), torch.argmax(pred[0], dim=0).detach().cpu().numpy())
                metric_validation.update(torch.argmax(pred, dim=1), y)
                AccuracyMetric_validation.update(torch.argmax(pred, dim=1), y)
                ConfusionMatrixValidation.update(torch.argmax(pred, dim=1), y)
        
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        print("[INFO EPOCH: {}/{}]".format(e +1, config.num_epochs))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

        if early_stopper.early_stop(avgTestLoss):             
            lr = lr*0.1
            if lr < 5*10e-10:
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
    
    torch.save(model, os.path.join(file_path, config.model))
    
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
    

