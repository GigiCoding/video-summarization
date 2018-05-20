-- This file testerC.lua works with trainerC.lua and LSTM_cumulative.lua
--	j_gate is a scalar (not tensor)
-- Current version works only for localization, not video-level classification... yet
-- Export outputs (prediction scores) to .csv file

local tester3 = {}
tester3.__index = tester3

function tester3:create()
    local self = {}
    setmetatable(self, tester)    
    return self
end

-- utility function:
function prepro(x,y)
-- input x: batch_size * inputsize * rho
-- input y: batch_size * nClasses * rho
    x = x:transpose(1,2):contiguous() -- swap axes 1,2 
    y = y:transpose(1,2):contiguous()
    if x:nDimension() == 3 then
	x = x:transpose(1,3):contiguous() -- swap axes 1,3
    end
    if y:nDimension() == 3 then
	y = y:transpose(1,3):contiguous()
    end
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    y = y:float():cuda()
-- x: rho*batchsize*inputsize
-- y: rho*batchsize*nclasses
    return x,y
end

function tester3:test()
-- outputs: predictions for each frame
-- gate_sums: for spasity computation
	-- local outputs = torch.Tensor(data.testFeats:size(1),opt.nClasses,opt.rho):fill(0)
  local predsFrames, tarFrames, numNZ, inputs, targets
  if opt.task == 'localization' then
    predsFrames = torch.Tensor(data.testFeats:size(1),opt.nClasses,opt.rho):cuda()
    tarFrames = torch.Tensor(data.testFeats:size(1),opt.nClasses,opt.rho):cuda()
    numNZ = torch.Tensor(torch.floor(data.testFeats:size(1)/opt.batch_size)):fill(0)
    -- Batch test:
    inputs = torch.Tensor(opt.batch_size, opt.input_size, opt.rho)
    targets = torch.Tensor(opt.batch_size, opt.nClasses, opt.rho)
  elseif opt.task == 'classification' then
    predsFrames = torch.Tensor(data.testFeats:size(1),opt.nClasses,opt.rho):cuda()
    tarFrames = torch.Tensor(data.testFeats:size(1),opt.nClasses):cuda()
    numNZ = torch.Tensor(torch.floor(data.testFeats:size(1)/opt.batch_size)):fill(0)
    -- Batch test:
    inputs = torch.Tensor(opt.batch_size, opt.input_size, opt.rho)
    targets = torch.Tensor(opt.batch_size, opt.nClasses)
  end
  local all_gates = torch.Tensor(data.testFeats:size(1),opt.rho):cuda()
    -- initialize the rnn state to all zeros
    init_state = {}
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size):double():cuda()
    -- local j_init = torch.zeros(opt.batch_size, 1)+0.0001    
    table.insert(init_state, h_init:clone())	-- for c
    table.insert(init_state, h_init:clone())	-- for h
-- We don't need to j_gate in current state, because no back prop. for testing
    current_state = init_state
    state_size = #current_state

    -- local vars
    local time = sys.clock()
    local timer = torch.Timer()
    local dataTimer = torch.Timer()	
    model:cuda()	
    model:evaluate()
    -- test over test data
    print(sys.COLORS.blue .. '========> testing on test set')
   
    for ptr = 1, data.testFeats:size(1), opt.batch_size do
	if ptr+opt.batch_size > data.testFeats:size(1) then break end
	local dataTime = dataTimer:time().real
    	xlua.progress(ptr,data.testFeats:size(1))
	
	-- create mini batch
	local idx = 1
	inputs:fill(0)	-- clear out for a new batch
	targets:fill(0)
	for i = ptr,ptr+opt.batch_size-1 do
	    if i <= data.testFeats:size(1) then
		inputs[idx] = data.testFeats[i]
		targets[idx] = data.testTargets[i]
		idx = idx + 1
	    end
	end
	x, y = prepro(inputs, targets, opt)
	-- forward pass	
	local numNZGates = 0	-- clear nonzero count
	-- make prediction for each of the images frames, start from frame #2
	for t = 1, opt.rho do
	    local lst = model:forward{x[t], unpack(current_state)}
	    -- lst = [c,h,x_acc,j_acc,j_gate,output]
	    local bound = ptr+opt.batch_size-1
	    -- if bound > data.testFeats:size(1) then bound = data.testFeats:size(1) end
	    all_gates[{{ptr,bound},{t}}]= lst[#lst-1]		--batchsize*rho
	    predsFrames[{{ptr,bound},{},{t}}]= lst[#lst]	--batchsize*nclass*rho
	    tarFrames[{{ptr,bound},{},{t}}] = y[t]
	    current_state = {}
	    for ss = 1,state_size do table.insert(current_state,lst[ss]) end
	    if lst[#lst-1]:nonzero():dim()>0 then 	-- lst[#lst-1]: j_gate
	        numNZGates = numNZGates + lst[#lst-1]:nonzero():size(1)
	    end
	end
     	numNZ[(ptr-1)/opt.batch_size+1] = numNZGates/opt.rho/opt.batch_size
print(numNZGates/opt.rho/opt.batch_size)	-- sparsity

    -- one batch is finished; clear out current states
	current_state = init_state
    end
    sparsity = 1-numNZ:mean()
    
    time = sys.clock()-time                                                              
    time = time/data.testFeats:size(1)
    print("\n ==> time to test 1 sample = " .. (time*1000) .. 'ms')
    timer:reset()
    dataTimer:reset()

print(sparsity)

-- save predictions:
    paths.mkdir(opt.pred_dir)
    --local savepreds = string.format('%s/predictions_LSTM_baseline.csv',opt.pred_dir)
    local savepreds = string.format('%s/predictions_%s_sparsity%f_lambda%f_threshold%f.csv', opt.pred_dir, opt.task, sparsity, opt.l1_weight, opt.thrsh)
    local savetargets = string.format('%s/targets.csv',opt.pred_dir)

print(sys.COLORS.green .. '---------> saving predictions...')
    local timer_sv = torch.Timer()
    local out1 = assert(io.open(savepreds,'w'))
    local preds = predsFrames:permute(1,3,2)	-- vids*rho*nClasses

    for i=1,preds:size(1) do 
	for j=1,preds:size(2) do 
	    for k=1,preds:size(3) do
-- print(preds[{{i},{j},{k}}])
		out1:write(preds[{{i},{j},{k}}]:squeeze())
		if k==preds:size(3) then out1:write('\n') else out1:write(',') end
	    end
	end
    end	
    out1:close()
print('... total time spent on saving predictions: ' .. timer_sv:time().real/60 .. ' min')
    local out2 = assert(io.open(savetargets,'w'))
    local tar = tarFrames:permute(1,3,2)
    for i=1,tar:size(1) do 
	for j=1,tar:size(2) do 
	    for k=1,tar:size(3) do
		out2:write(tar[{{i},{j},{k}}]:squeeze())
		if k==tar:size(3) then out2:write('\n') else out2:write(',') end
	    end
	end
    end	
print('Done!')

--save test gates:
    if opt.save_testgates == true then 
	paths.mkdir(opt.gate_dir)
	local savegates = string.format('%s/gate_%s_lambda_%f_threshold%f.csv', opt.gate_dir, opt.task,  opt.l1_weight,opt.thrsh)
	print('saving test gates to '.. savegates)
	--[[local saver = require 'util.save_gates'
	saver:create(true) -- quiet mode
	saver:save(all_gates,',',savegates)
	--saver:save(gate_sums,',',savegates)
	]]--
-- all_gates:video*rho
	local out3 = assert(io.open(savegates,'w'))
	for i=1,all_gates:size(1) do
	    for j=1,all_gates:size(2) do
	    	out3:write(all_gates[i][j])
		if j==all_gates:size(2) then out3:write('\n') else out3:write(' ') end
	    end
	end
    	out3:close()
    end

end

return tester3





























