-- Guangxiao Zhang H., Jan 2018
require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'lfs'
require 'optim'
require 'sys'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'
require 'pastalog'

local opts = require 'opts'
opt = opts.parse(arg)
print(sys.COLORS.blue .. '==========================================================')
print(sys.COLORS.blue .. '		Charades Activity Dataset		    ')
print(sys.COLORS.blue .. '==========================================================')
print('Extended LSTM gates for action recogntion and key frame selection')
print('				    Guangxiao Zhang @umd, Jan 2018')
-- print(opt)

-- print(sys.COLORS.red .. 'lambda='..opt.l1_weight..';   test threshold='..opt.thrsh..'; ')
print(sys.COLORS.red .. 'learning rate='..opt.learning_rate)

-- initialize cunn/cutorch for training on the GPU 
print( 'using CUDA on GPU ' .. opt.gpuid .. '...')
local cutorch = require 'cutorch'
cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
cutorch.manualSeed(opt.seed)

-- load data
print(sys.COLORS.green .. '-------- > Loading videos...')
local datafile
if opt.task == 'localization' then datafile = 'rgbfeats_157classes_50frames.t7'
elseif opt.task == 'classification' then datafile = 'rgbfeats_videolabels_50frames8FPS.t7' end
data = torch.load(opt.dir_feats .. datafile)
-- data = {trainFeats, trainTargets, testFeats, testTargets}
print('Done loading videos!')
print(data)
print(sys.COLORS.green .. '-------- > Creating batches...')
local BatchLoader = require 'util.BatchLoader'
loader = BatchLoader:create(data,opt.batch_size,opt.input_size,opt.rho,opt.nClasses)

if opt.test_only then
    --test
    print(sys.COLORS.green .. '-------- > Loading checkpoint...')
    local checkpoint = torch.load('checkpoint/'.. opt.resume_from)
    model = checkpoint.protos.rnn
else
    -- train
    print(sys.COLORS.green .. '-------- > Preparing trainer...')
    local trn_lstm
    if opt.model == 'LSTM_test2' then trn_lstm = require 'trainer2'
    elseif opt.model == 'LSTM_test3' then trn_lstm = require 'trainer3'
    elseif opt.model == 'LSTM_cumulative' then trn_lstm = require 'trainerC'
    elseif opt.model == 'LSTM_binaryJ' then trn_lstm = require 'trainer_binaryJ'
    elseif opt.model == 'LSTM_topK' then trn_lstm = require 'trainer_topK2'
    else trn_lstm = require 'trainer' end
    trn_lstm:create()
    local trn = trn_lstm:train()
    model = trn.protos.rnn
end
--[[
print(sys.COLORS.green .. '-------- > Testing...')
local tst_lstm
if opt.model == 'LSTM_test2' then tst_lstm = require 'tester2'
elseif opt.model == 'LSTM_test3' then tst_lstm = require 'tester3'
elseif opt.model == 'LSTM_cumulative' then tst_lstm = require 'testerC'
elseif opt.model == 'LSTM_binaryJ' then tst_lstm = require 'tester_binaryJ'
elseif opt.model == 'LSTM_topK' then tst_lstm = require 'tester_topK'
else tst_lstm = require 'tester' end
	-- warning: tester2 has bugs... need to be fixed!!
tst_lstm:create()
tst_lstm:test()
]]--




