--[[

Train two parallel LSTM networks, one regular LSTMs,
and one with L1-norm constraint on in_gate.
Test on UCF-101

This code is based on Karpathy char-rnn:
https://github.com/karpathy/char-rnn

Guangxiao Zhang, Jan 2017

]]--

require 'torch'
require 'cunn'
require 'nngraph'
nn = require 'nn'
sys = require 'sys'
xlua = require 'xlua'    -- xlua provides useful tools, like progress bars
optim = require 'optim'
pastalog = require 'pastalog'
require 'lfs'
require 'optim'
require 'lfs'
require 'util.misc'

model_utils = require 'util.model_utils'
LSTM = require 'model.LSTM'
LSTM_gateOut = require 'model.LSTM_gateOut_offset'
require 'model.L1Criterion'

cmd = torch.CmdLine()
cmd:text('Options')
-- data prepartion
cmd:option('-spatFeatDir','/media/storage/Work/data/UCF101_features/','directory of spatial feature')
cmd:option('-tempFeatDir','none','directory of temporal feature from optical flow')
cmd:option('-dataset','ucf101','database')
cmd:option('-split', 1, 'split set index (1|2|3)')
cmd:option('-num_segment', 1, 'number of segments for each video')
cmd:option('-input_size',2048, 'dimension of input features')
cmd:option('-rho', 25, 'number of frames for each video')
-- model params
cmd:option('-lstm_inGate', true, 'output lstm in_gate, true or false')
cmd:option('-rnn_size', 1024, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-l1_weight',0.001, 'l1 regularization penalty')
cmd:option('-offset',-0.01,'small offset for in_gates to push sigmoid function to zero')
-- optimization
cmd:option('-learning_rate',1e-5,'learning rate')
cmd:option('-learning_rate_decay',0.1,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-decay_every',20, 'number of iterations decay rate drops')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
--cmd:option('-seq_length',64,'number of timesteps to unroll for')
cmd:option('-batch_size',64,'number of sequences to train on in parallel') -- number of examples per batch
cmd:option('-max_epochs',75,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generating seed')
cmd:option('-print_every',50,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',100,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', '/usr/local/src/lstm_l1/checkpoint', 'output directory where checkpoints get written')
cmd:option('-savecheckpoint',false,'run the training code without saving the checkpoint')
cmd:option('-savefile','lstm_l1','filename to autosave the checkpoint to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')

--test options
cmd:option('-test_only',false,'run the test part only, using model from option resume_from')
cmd:option('-resume_from','checkpoint/lm_lstm_l1_epoch75.00_0.0604.t7','model checkpoint to use for testing')
cmd:option('-averagePred',false,'average prediction')
cmd:option('-save_test', '/usr/local/src/lstm_l1/test_results','directory to save the test results.')
cmd:option('-save_gates','/usr/local/src/lstm_l1/intermediate_results_gateshifted','directory to save the gate values')
cmd:option('-pastalogName', 'model_lstm', 'the name of your experiment')
cmd:option('-print_confusion',true, 'print out the confusion table or be silent')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

print('***** Running sparse gate lstm *****')
print('lambda = '.. opt.l1_weight)
print('learning_rate = '.. opt.learning_rate)

opt.save = 'testlog_' .. opt.pastalogName
path.mkdir(opt.save)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- load data
opt.spatial = paths.dirp(opt.spatFeatDir) and true or false
opt.temporal = paths.dirp(opt.tempFeatDir) and true or false

data = require 'util.video_loader'
-- data = {trainData, trainTarget, testData, testTarget, nClass}
classes = require 'datainfo.ucf-101'
BatchLoader = require 'util.BatchLoader'
loader = BatchLoader:create(data,opt.batch_size,opt.input_size,opt.rho)

print('Database: '..opt.dataset)
print('frame length per video: '..opt.rho)
--[[print('size of training set:')
print(data.trainData:size())
print('size of testing set:')
print(data.testData:size())]]--

if opt.test_only == true then
	print(sys.COLORS.green ..'====> loading pre-trained LSTM model...')	
	-- load model checkpoint
	local checkpoint = torch.load(opt.resume_from)
--print(checkpoint)
	local protos = checkpoint.protos
	model = protos.rnn
	criterion = protos.criterion
else
	-- train LSTM mode
	local trn_lstm = require 'train_lstm_l1'
	trn_lstm:create()
	local trn = trn_lstm:train()
	local protos = trn.protos	
	model = protos.rnn
	criterion = protos.criterion
end

-- test
local tst = require 'test_lstm'
tst:create()
tst:test(data.testData, data.testTarget, model, criterion,opt)








