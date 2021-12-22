function [ net ] = get_test(varargin)

addpath('utilities');
opts.idx_gpus =0; % 0: cpu
opts.matconvnet_path =  'D:/anzhuang/matlab/matconvnet-1.0-beta25/matlab/vl_setupnn.m';
opts.net_path = 'E:/yanjiu/chengxu/AMGAN/新建文件夹/model/net.mat'; 

opts = vl_argparse(opts, varargin);
run(opts.matconvnet_path);

%% load network
net = load(opts.net_path);
net = net.net(1); % idx 1: Generator, 2: Discriminator
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
if opts.idx_gpus >0
%     gpuDevice()
    net.move('gpu');
end
rng('default')
input1=importdata('noisy1.mat');
input1= single(input1);
if opts.idx_gpus >0,   input1 = gpuArray(input1);    end
net.eval({'input',input1});
im_out1 = net.vars(net.getVarIndex('prediction1')).value;
quzao1=im_out1;
if opts.idx_gpus
    quzao1 = gather(quzao1);
    input1  = gather(input1);
end
input2=importdata('noisy2.mat');
input2= single(input2);
if opts.idx_gpus >0,   input2 = gpuArray(input2);    end
net.eval({'input',input2});
im_out2= net.vars(net.getVarIndex('prediction1')).value;
quzao2=im_out2;
if opts.idx_gpus
    quzao2 = gather(quzao2);
    input2  = gather(input2);
end
input3=importdata('noisy3.mat');
input3= single(input3);
if opts.idx_gpus >0,   input3 = gpuArray(input3);    end
net.eval({'input',input3});
im_out3 = net.vars(net.getVarIndex('prediction1')).value;
quzao3=im_out3;
if opts.idx_gpus
    quzao3 = gather(quzao3);
    input3  = gather(input3);
end
quzao=[quzao1;quzao2;quzao3];
chunjing=importdata('pure.mat');
hanzao=importdata('noisy.mat');
chazhi=hanzao-quzao;
figure(1)%纯净信号
wiggle(chunjing);title('pure data');xlabel('Trace number');ylabel('Times(ms)');
figure(2)%含噪信号
wiggle(hanzao);title('noisy data'); xlabel('Trace number');ylabel('Times(ms)');
figure(3)%去噪信号
wiggle(quzao);title('denoising data');xlabel('Trace number');ylabel('Times(ms)');
figure(4)%噪声
wiggle(hanzao-chunjing);title('noise data');xlabel('Trace number');ylabel('Times(ms)');
figure(5)%差值
wiggle(chazhi);title('difference data');xlabel('Trace number');ylabel('Times(ms)');
figure(6)
daoshu=800;
plot(hanzao(:,daoshu),'g');axis([500 2000 -4 4])
hold on
plot(quzao(:,daoshu),'k','LineWidth',1.3)
hold on
plot(chunjing(:,daoshu),':ro')
dx=0.0004;
[S1,f1,k1] = fk(chunjing,dx,5,9);                                %含噪信号
figure,imagesc(k1,f1,S1);axis([-0.1 0.1 0 150]);title('pure data');xlabel('k[c/m]');ylabel('f(Hz)');                     
[S2,f2,k2] = fk(hanzao,dx,5,9);                                %含噪信号
figure,imagesc(k2,f2,S2);title('noisy data');axis([-0.1 0.1 0 150]);xlabel('k[c/m]');ylabel('f(Hz)');
[S3,f3,k3] = fk(quzao,dx,5,9);                                %去噪信号
figure,imagesc(k3,f3,S3);title('denoising data');axis([-0.1 0.1 0 150]);xlabel('k[c/m]');ylabel('f(Hz)');
[S4,f4,k4] = fk(chazhi,dx,5,9);                                %差值信号
figure,imagesc(k4,f4,S4);title('difference data');axis([-0.1 0.1 0 150]);xlabel('k[c/m]');ylabel('f(Hz)');
[S5,f5,k5] = fk(hanzao-chunjing,dx,5,9);                                %噪声
figure,imagesc(k5,f5,S5);title('noise data');axis([-0.1 0.1 0 150]);xlabel('k[c/m]');ylabel('f(Hz)');
snr = SNR_singlech(chunjing,hanzao);
disp(['SNR before denoising ： ',num2str(snr)]);       
snrthen = SNR_singlech(chunjing,quzao);
disp(['SNR after denoising ： ',num2str(snrthen)]);
chunjing=single(chunjing);
hanzao=single(hanzao);
mse_yuanshi=immse(hanzao,chunjing);
disp(['MSE before denoising： ',num2str(mse_yuanshi)]); 
mse=immse(quzao,chunjing);
disp(['MSE after denoising： ',num2str(mse)]);
