from __future__ import print_function, division
import argparse

from data_loader.Load_RLT_face_audio_OpenFace_Affect import *

from data_loader import *
from models.fusion_net import FusionModule, FusionModule_ours
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import cv2
from utils import AvgrageMeter, performances, performances_ours
import torch.utils.data


def set_seed(seed=40):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(40)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(40)

# feature  -->   [ batch, channel, height, width ]
def get_train_dataset_loader(args):
    if args.train_dataset == "RLT":
        train_data = getattr(Load_RLT_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "BagOfLies":
        train_data = getattr(Load_BagOfLies_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "BoxOfLies":
        train_data = getattr(Load_BoxOfLies_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    elif args.train_dataset == "MU3D":
        train_data = getattr(Load_MU3D_face_audio_OpenFace_Affect, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    # stage 2 data loader
    elif args.train_dataset == "DOLOS":
        train_data = getattr(Load_DOLOS_data, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))

    elif args.train_dataset == "MDPE":
        train_data = getattr(Load_MDPE_data, args.train_dataset + '_train')(
            args.train_list, transform=transforms.Compose([RandomHorizontalFlip(),
                                                                            ToTensor(),
                                                                            Normaliztion()]))
    else:
        raise Exception("Train dataset name not exists!")
        # train_data = None

    return train_data


def get_test_dataset_loader(args):
    if args.test_dataset == "RLT":
        test_data = getattr(Load_RLT_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    elif args.test_dataset == "BagOfLies":
        test_data = getattr(Load_BagOfLies_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    elif args.test_dataset == "BoxOfLies":
        test_data= getattr(Load_BoxOfLies_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    elif args.test_dataset == "MU3D":
        test_data = getattr(Load_MU3D_face_audio_OpenFace_Affect, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    elif args.test_dataset == "DOLOS":
        test_data = getattr(Load_DOLOS_data, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    elif args.test_dataset == "MDPE":
        test_data = getattr(Load_MDPE_data, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion(), ToTensor_test()]))
    elif args.test_dataset == "MMDD":
        test_data = getattr(Load_MMDD_data, args.test_dataset + '_test')(
            args.test_list, transform=transforms.Compose([Normaliztion_mmdd(), ToTensor_test_mmdd()]))
    else:
        raise Exception("Test dataset name not exists!")
        # test_data = None
    return test_data


def FeatureMap2Heatmap(x, x2):
    ## initial images
    ## initial images
    org_img = x[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log + '/' + args.log + '_visual.jpg', org_img)

    org_img = x2[0, :, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))
    # org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log + '/' + args.log + '_audio.jpg', org_img)


# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(
        args.log + '/' + args.fusion_type + "_" + args.modalities + "_" + args.train_dataset + "_to_" +
        args.test_dataset + str(args.test_list.split('/')[-1].split('.')[0]) + '_fusion_test.txt',
        'w')

    echo_batches = args.echo_batches

    print("Deception Detection!!!:\n ")

    log_file.write('Deception Detection!!!:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?

    print('train from scratch!\n')
    log_file.write('train from scratch!\n')
    log_file.flush()

    # model = ResNet18_GRU(pretrained=True, GRU_layers=1)
    # model = ResNet18_BiGRU(pretrained=True, GRU_layers=2)
    # model = ResNet18(pretrained=True)
    model = FusionModule_ours(args)  # 整个的模型

    # # model = OpenFaceAU_MLP_MLP()
    # # model = OpenFaceGaze_MLP_MLP()
    # # model = OpenFaceGaze_AllMLP()

    model = model.cuda()
    # model = model.to(device[0])
    # model = nn.DataParallel(model, device_ids=device, output_device=device[0])
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print(model)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    
    # 初始化变量，用于记录当前最佳权重文件路径
    best_checkpoint_path = None
    best_test_filename_upload = None


    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()  # possible warining that scheduler.step is before optimizer.step()---> should be after it!!!
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_absolute = AvgrageMeter()
        loss_contra = AvgrageMeter()
        loss_absolute_RGB = AvgrageMeter()
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        train_data = get_train_dataset_loader(args)
     
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs = sample_batched['video_x'].cuda()  # [16, 3, 32, 224, 224]  32帧图像
            inputs_audio = sample_batched['audio_x'].cuda()  # [16, 3, 224, 224] 语谱图特征，最后一个音波的谱不用管1维的结果
            if inputs_audio.shape[1] == 4:
                inputs_audio = inputs_audio[:,  :3, :, :]
            # print('inputs_audio', inputs_audio[:, :3, :, :])
            # print('inputs_wave', inputs_audio[:, 3, :, :])

            inputs_OpenFace = sample_batched['OpenFace_x'].cuda()  # [16, 43, 64] openface特征
            inputs_affect = sample_batched['x_affect'].cuda()  # [16, 7, 64] 情感效应 of openface特征
            
            # print('inputs:', inputs.size(), inputs_audio.size(), inputs_OpenFace.size(), inputs_affect.size())

            DD_label = sample_batched['DD_label'].cuda()

            # print('DD_label:', DD_label.size())  # [16, 1] 真假判别标签


            inputs_OpenFace = torch.cat((inputs_affect, inputs_OpenFace), dim=1)  # [16, 50, 64] 合并openface  

            # join affect feature to openface  feature

            optimizer.zero_grad()

            log_file.write('\n')
            log_file.flush()

            
            # vision, audio, face
            if args.fusion_type != 'crosstrans':
                v_logit, a_logit, f_logit, fused_logit = model(inputs_OpenFace, inputs_audio, inputs)  # openface特征[b, 50, 64]，语谱图特征[b, 3, 224, 224]，视频图像[b, 3, 32, 224, 224]
            else:
                v_logit, a_logit, f_logit, fused_logit = model(inputs_OpenFace, inputs_audio, inputs)  # openface特征[b, 50, 64]，语谱图特征[b, 3, 224, 224]，视频图像[b, 3, 32, 224, 224]

            loss_global = criterion(fused_logit , DD_label.squeeze(-1)) * args.fused_weight  # 0.5也是一个超参数可以进行调整
            if 'v' in args.modalities and v_logit is not None:
                v_loss = criterion(v_logit, DD_label.squeeze(-1))
                loss_global += v_loss
            if 'a' in args.modalities and a_logit is not None:
                a_loss = criterion(a_logit, DD_label.squeeze(-1))
                loss_global += a_loss
            if 'f' in args.modalities and f_logit is not None:
                f_loss = criterion(f_logit , DD_label.squeeze(-1))
                loss_global += f_loss

            loss = loss_global

            loss.backward()

            optimizer.step()

            n = inputs.size(0)
            loss_absolute.update(loss_global.data, n)
            loss_contra.update(loss_global.data, n)
            loss_absolute_RGB.update(loss_global.data, n)

            if i % echo_batches == echo_batches - 1:  # print every 50 mini-batches

                # visualization
                FeatureMap2Heatmap(inputs[:, :, 1, :, :], inputs_audio)  #  第一帧图像 [bs=16, 3, 224, 224] + [16, 3, 224, 224] 语谱图特征

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE1= %.4f , CE2= %.4f \n' % (
                    epoch + 1, i + 1, lr, loss_absolute.avg, loss_contra.avg, loss_absolute_RGB.avg))

        # whole epoch average
        log_file.write('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE1= %.4f , CE2= %.4f \n' % (
            epoch + 1, i + 1, lr, loss_absolute.avg, loss_contra.avg, loss_absolute_RGB.avg))
        log_file.flush()

        epoch_test = 1

        if epoch % epoch_test == epoch_test - 1:  # test every 5 epochs
            model.eval()

            with torch.no_grad():

                ###########################################
                '''                test             '''
                ##########################################
                # # differenet clip num for each video, it cannot stack into large batachsize, set = 1
                test_data = get_test_dataset_loader(args)

                dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

                map_score_list = []
                final_score_list = []

                for i, (sample_batched, videoname) in enumerate(dataloader_test):

                    inputs = sample_batched['video_x'].cuda()  # [1, 1, 3, 32, 224, 224]

                    inputs_OpenFace = sample_batched['OpenFace_x'].cuda()  # [1, 1, 43, 64]
                    inputs_audio = sample_batched['audio_x'].cuda()  # [1, 1, 3, 224, 224]
                    if inputs_audio.shape[2] == 4:
                        inputs_audio = inputs_audio[:, :, :3, :, :]
                    inputs_affect = sample_batched['x_affect'].cuda()  # [1, 1, 7, 64]
                    print('name:', videoname[0])

                    optimizer.zero_grad()

                    for clip_t in range(inputs_OpenFace.shape[1]):
                        _, _, _, fused_logit = model(
                            torch.cat((inputs_affect[:, clip_t, :, :], inputs_OpenFace[:, clip_t, :, :]), dim=1),
                            inputs_audio[:, clip_t, :, :, :],
                            inputs[:, clip_t, :, :, :, :])

                        if args.fusion:
                            if clip_t == 0:
                                logits_accumulate = F.softmax(fused_logit, -1)
                            else:
                                logits_accumulate += F.softmax(fused_logit, -1)
                        else:
                            raise Exception("testing for fusion only!")
                        # todo: test for other single modality models with/without fusion
                    logits_accumulate = logits_accumulate / inputs_OpenFace.shape[1]
                    for test_batch in range(inputs_audio.shape[0]):
                        # # 原来的，为了计算用
                        # map_score_list.append(
                        #     '{} {}\n'.format(logits_accumulate[test_batch][1], DD_label[test_batch][0]))

                        # 为了提交使用
                        final_score_list.append(
                            '{} {}\n'.format(videoname[0], logits_accumulate[test_batch][1]))
                        

                    
                # 保存最新的
                print('Current epoch:', epoch+1)
                if epoch == args.epochs-1:
                    if not os.path.exists("./ckpt_{}_{}_{}/checkpoint/".format(str(args.batchsize), str(args.epochs), str(args.fused_weight))):
                        os.makedirs("./ckpt_{}_{}_{}/checkpoint/".format(str(args.batchsize), str(args.epochs), str(args.fused_weight)))
                        
                    best_checkpoint_path = "./ckpt_{}_{}_{}/checkpoint/{}_{}_{}_{}_ep{}.pt".format(str(args.batchsize), str(args.epochs), str(args.fused_weight), args.fusion_type, args.modalities, args.train_dataset, args.test_dataset, str(epoch+1))

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_val_acc,
                    }, best_checkpoint_path)

                    best_test_filename_upload = args.log + '/' + args.fusion_type + "_" + args.modalities + "_" + \
                                    args.train_dataset + "_to_" + args.test_dataset + "_" + \
                                    str(args.test_list.split('/')[-1].split('.')[0]) + str(epoch+1) + "_upload.txt"
                    with open(best_test_filename_upload, 'w') as file:
                        file.writelines(final_score_list)

                log_file.flush()

    print('Finished Training')
    log_file.close()


if __name__ == "__main__":

    # set gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--device', type=int, default=0, help='the gpu id used for predict')
    # parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  # default=0.001
    parser.add_argument('--batchsize', type=int, default=16, help='initial batchsize')  # 16
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 20
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='gamma of optim.lr_scheduler.StepLR, decay of lr')  # 0.1
    parser.add_argument('--echo_batches', type=int, default=1, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=30, help='total training epochs')
    parser.add_argument('--log', type=str, default="logsintra", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    # dataset dirs
    # parser.add_argument('--train_dataset', type=str, default='MDPE')
    # parser.add_argument('--train_root', type=str, default='',
    #                     help='train dataset root dir')
    # parser.add_argument('--train_list', type=str, default='./dataset2/MDPE_train_balanced_l493_t492.pkl',
    #                     help='train feature list')
    parser.add_argument('--train_dataset', type=str, default='DOLOS')
    parser.add_argument('--train_root', type=str, default='',
                        help='train dataset root dir')
    parser.add_argument('--train_list', type=str, default='./dataset2/DOLOS_train_l464_t365.pkl',
                        help='train feature list')

    parser.add_argument('--test_dataset', type=str, default='MMDD')
    parser.add_argument('--test_root', type=str, default='',
                        help='test data root')
    parser.add_argument('--test_list', type=str, default='./dataset2/MMDD_test_features.pkl',
                        help='test feature list')

    # config for individual modal
    parser.add_argument('--fusion', action='store_true', default='false', help='true when fusion module is used')
    parser.add_argument('--fusion_modal', type=str, default='FusionModule')

    parser.add_argument('--modalities', type=str, default='vaf', help='modalities in v-affect+openface, a-audio, '
                                                                      'f-face frames')

    parser.add_argument('--v_model', type=str, default='OpenFace_Affect7_MLP_MLP_ours')  # OpenFace_Affect7_MLP_MLP
    parser.add_argument('--a_model', type=str, default='ResNet18_audio')  # ResNet18_audio
    parser.add_argument('--f_model', type=str, default='ResNet18_GRU')  # ResNet18_GRU

    # dimensions for each modality (the embedding size)
    parser.add_argument('--v_dim', type=int, default=64)
    parser.add_argument('--a_dim', type=int, default=512)
    parser.add_argument('--f_dim', type=int, default=256)

    # train with fusion parameters
    parser.add_argument('--fusion_type', type=str, default='mix_concat', help='modality fusion type in '
                                                                          'concat/transformer/senet/mix_concat/crosstrans')
    parser.add_argument('--concat_dim', type=int, default=-1, help='concatenation dim for concat fusion method')
    # config for transformer
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='attention dropout (for audio)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--layers', type=int, default=2,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
    parser.add_argument('--fused_weight', type=float, default=0.5)
    
    # config for senet
    parser.add_argument('--channel', type=int, default=64, help='channel dimension for linear layer')
    parser.add_argument('--reduction', type=int, default=16, help='linear dimension reduction')

    args = parser.parse_args()
    train_test()
