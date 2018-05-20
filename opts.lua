-- 
local M = {}

function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text('Options:')
	cmd:option('-exp_name',		'0426_2348', 'experiment name for saving data')
	-- data prepartion
	cmd:option('-dataset',		'charades','database')
	cmd:option('-task',		'localization', 'task (classification|localization)')
	cmd:option('-nClasses',		157, 'number of classes (including background)')
	cmd:option('-dir_feats',	'/media/storage/Work/data/Charades_features/', 'Path to features')
	--cmd:option('-chunk_path',	'chunk_rgb50f_training/', 'the subfolder under dir_feats that saves features in chunks')
	--cmd:option('-rgbt7',		'rgbfeats_157classes_50frames.t7','t7 file of all features')
	--cmd:option('-rgbt7Train',	'rgbfeats_train_50frames_8FPS.t7', 't7 file containing rgb features for training')
	--cmd:option('-rgbt7Test',	'rgbfeats_test_50frames_8FPS.t7', 't7 file containing rgb features for testing')
	--cmd:option('-rgbfeats',	'Charades_v1_features_rgb/', 'Path to rgb features')
	--cmd:option('-flowfeats',	'Charades_v1_features_flow/', 'Path to flow features')
   	--cmd:option('-dir_annos',	'/media/storage/Work/data/Charades_annotations', 'Path to annotations')
	--cmd:option('-trainfile',  	'./Charades_v1_train.csv', 'Path to training annotations')
   	--cmd:option('-testfile',   	'./Charades_v1_test.csv', 'Path to testing annotations')
	--cmd:option('-num_segment', 	1, 'number of segments for each video')
	cmd:option('-input_size',	4096, 'dimension of input features')
	cmd:option('-rho', 		50, 'number of frames for each video')
	cmd:option('-batch_size',	64,'number of sequences to train on in parallel') -- number of examples per batch
	--cmd:option('-chuck_size',	1000, 'number of videos to load at once')
	-- model params
	cmd:option('-model',		'LSTM_test','LSTM model (LSTM|LSTM_gateOut|LSTM_gateOut_offset|LSTM_gateOut2|LSTM_extended)')
	cmd:option('-Kframes',		'10','top K frames to keep in the summary')
	cmd:option('-baseline',		false, 'set LSTM to baseline mode with j=1 (true|false)')
	cmd:option('-rnn_size', 	512, 'size of LSTM internal state')
	cmd:option('-num_layers', 	1, 'number of layers in the LSTM')
	cmd:option('-l1_weight',	0, 'weight on regularization penalty')
	cmd:option('-offset',		0,'small offset to push in_gate to zero.')
	cmd:option('-thrsh',		0.1,'threshold to set j_gate elements to zero, if sum(j_gate)<thrsh.')
	-- optimization
	cmd:option('-learning_rate',	1e-4,'learning rate')
	cmd:option('-learning_rate_decay',0.1,'learning rate decay')
	cmd:option('-learning_rate_decay_after',500,'in number of epochs, when to start decaying the learning rate')
	cmd:option('-decay_rate',	0.95,'decay rate for rmsprop')
	cmd:option('-decay_every',	100, 'number of iterations decay rate drops')
	cmd:option('-dropout',		0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
	--cmd:option('-seq_length',	64,'number of timesteps to unroll for')
	cmd:option('-max_epochs',	300,'number of full passes through the training data')
	cmd:option('-grad_clip',	5,'clip gradients at this value')
		    -- test_frac will be computed as (1 - train_frac - val_frac)
	cmd:option('-init_from', 	'', 'initialize network parameters from checkpoint at this path')
	-- bookkeeping
	cmd:option('-seed',		123,'torch manual random number generating seed')
	cmd:option('-print_every',	50,'how many steps/minibatches between printing out the loss')
	cmd:option('-eval_val_every',	1000,'every how many iterations should we evaluate on validation data?')
	cmd:option('-eval_shuffle',	false, 'shuffle or not in evaluation')
	cmd:option('-checkpoint_dir', 	'./checkpoint', 'output directory where checkpoints get written')
	cmd:option('-savecheckpoint',	false,'run the training code and save the checkpoint')
	-- cmd:option('-checkpointfn',	'lstm_v3','filename to autosave the checkpoint to. Will be inside checkpoint_dir/')
	cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
	-- GPU/CPU
	cmd:option('-gpuid',		0,'which gpu to use. -1 = use CPU')
	cmd:option('-opencl',		0,'use OpenCL (instead of CUDA)')

	--test options
	cmd:option('-test_only',	false,'run the test part only, using model from option resume_from (true|false)')
	cmd:option('-resume_from',	'checkpoint/lstm_baseline.t7','model checkpoint to use for testing')
	cmd:option('-averagePred',	false,'average prediction')
	--cmd:option('-save_testpreds', 	false, 'save the predictions or not (true|false)')
	cmd:option('-save_testgates', 	false, 'save j_gate values in test or not');
	cmd:option('-gate_dir',		'results/test_gates','directory that saves the gates in test.')
	cmd:option('-pred_dir',		'results/predictions','directory to save the predictions.')
	--cmd:option('-save_dir','./debug','directory to save the gate values')
	cmd:option('-pastalogName', 	'model_lstm', 'the name of your experiment')
	cmd:option('-print_confusion',	false, 'print out the confusion table or be silent')
	cmd:text()

	-- parse input params
	opt = cmd:parse(arg)
	opt.save = 'results/testlog_' .. opt.pastalogName
	paths.mkdir(opt.save)
	paths.mkdir(opt.checkpoint_dir)
	--opt.save_gates = paths.concat(opt.save_dir..'/results_max_lossv3_lambda'..opt.l1_weight..'_offset'..-1*opt.offset..'/')
	--print(opt.save_gates)
	--paths.mkdir(opt.save_gates)

	--opt.spatFeatDir = opt.dir_feats .. opt.rgbt7
	--opt.tempFeatDir = opt.dir_feats .. opt.flowfeats

	return opt
end

return M
