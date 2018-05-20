-- Guangxiao Zhang H. Jan 2018
-- Trainer for Charades, with j_gates being scalars instead of tensors
-- Note: classification task is undone yet. 

local LSTM_test3 	= require 'model.LSTM_test3'
local model_utils 	= require 'util.model_utils'
require 'util.misc'

local trainer = {}
trainer.__index = trainer

function trainer:create()
    local self = {}
    setmetatable(self, trainer)
    print('Creating an LSTM model with j_gate output...')
    protos = {}
    protos.rnn = LSTM_test3.lstm(opt.input_size, opt.rnn_size, 1, opt.nClasses, opt.dropout)  
    print('Initializing LSTM...')

    init_state = {}
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)     
    --local j_init = torch.zeros(opt.batch_size, 1)+0.0001
    h_init = h_init:cuda() 
    --j_init = j_init:cuda() 
	    
    table.insert(init_state, h_init:clone())	--GX: for next_c
    table.insert(init_state, h_init:clone())	--GX: for next_h 	
    --table.insert(init_state,j_init:clone())	--GX: for j_gates
    
    -- ship the model to GPU
    for k,v in pairs(protos) do v:cuda() end
    -- combine all parameters to flattened tensors
    params, grad_params = model_utils.combine_all_parameters(protos.rnn)

    local layer_idx = 1
    for _,node in ipairs(protos.rnn.forwardnodes) do
	if node.data.annotations.name == "i2h_" .. layer_idx then
	    print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
	    -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
	    node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
	end
    end
    print(sys.COLORS.blue .. 'number of parameters in the model: ' .. params:nElement())

    -- make a bunch of clones after flattening
    clones = {}
    for name,proto in pairs(protos) do 
	print('cloning ' .. name)
	clones[name] = model_utils.clone_many_times(proto, opt.rho, not proto.parameters)
    end

    self.protos = protos
    return self
end

-- process help functions 
function prepro(x,y)
-- input x: batch_size * inputsize * rho
-- input y: batch_size * nClasses (* rho)
    x = x:transpose(1,2):contiguous() -- swap axes 1,2 
    y = y:transpose(1,2):contiguous()
    if x:nDimension() == 3 then
	x = x:transpose(1,3):contiguous() -- swap axes 1,3
    end
    if y:nDimension() == 3 then
	y = y:transpose(1,3):contiguous()
-- y: rho*batchsize*nclasses, if y is frame level target; otherwise, y: batch_size*nClasses
    end
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    y = y:float():cuda()
-- x: rho*batchsize*inputsize

    return x,y
end

function eval_split(split_index, max_batches)
    epoch = epoch or 1
    local inputs,targets
    if opt.task == 'localization' then
        inputs = torch.Tensor(opt.batch_size, opt.input_size, opt.rho):cuda()
        targets = torch.Tensor(opt.batch_size, opt.nClasses, opt.rho):cuda()
    elseif opt.task == 'classification' then
	inputs = torch.Tensor(opt.batch_size, opt.input_size, opt.rho):cuda()
        targets = torch.Tensor(opt.batch_size, opt.nClasses):cuda()
    end
    -- shuffle at each epoch:
    local shuffle = torch.randperm(data.trainFeats:size(1))
    local rnn_state = {[0] = init_state}
    local loss = 0
    local loss_max = 0 
    local loss_cls = 0
    local numNZGates = 0
    local numNZ = torch.Tensor(torch.floor(data.trainFeats:size(1)/opt.batch_size))
    local zerotensors = torch.Tensor(opt.batch_size):zero():cuda() -- ???

    -- iterate over batches in the split
    for ptr = 1,data.trainFeats:size(1),opt.batch_size do
	-- load a mini batch, if not exceeding the size
	if (ptr+opt.batch_size-1)>data.trainFeats:size(1) then break end	
	local idx = 1	-- index within mini batch 
	for i=ptr,ptr+opt.batch_size-1 do 
	    inputs[idx] = data.trainFeats[shuffle[i]]
	    targets[idx] = data.trainTargets[shuffle[i]]
	    idx = idx+1
	end
	local x,y = prepro(inputs, targets)
	
	local predictions = {}
	local frame_wt = {}
	all_jgates = {}
	-------- foraward pass ----------
      if opt.task == 'localization' then
	for t=1,opt.rho do 
	    local prev_c, prev_h = unpack(rnn_state[t-1])	
	    local lst = clones.rnn[t]:forward{x[t],prev_c,prev_h}
	    rnn_state[t]={}
	    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
	    table.insert(all_jgates, lst[#lst-1])		-- the second last is j_gate
            table.insert(predictions,lst[#lst]) 
	    
	    local abscrit = nn.AbsCriterion():cuda()
	    local clscrit = nn.BCECriterion():cuda()
	    local loss1 = abscrit:forward(all_jgates[t], zerotensors)
	    local loss2 = clscrit:forward(predictions[t],y[t])
	    loss_max = loss_max + loss1
	    loss_cls = loss_cls + loss2
	    if all_jgates[t]:nonzero():dim()>0 then 	-- it might be 'nil', if all j_gate = 0
	        numNZGates = numNZGates + all_jgates[t]:nonzero():size(1)
	    end
	end
      elseif opt.task == 'classification' then
	for t=1,opt.rho do 
	    local prev_c, prev_h = unpack(rnn_state[t-1])	
	    local lst = clones.rnn[t]:forward{x[t],prev_c,prev_h}
	    rnn_state[t]={}
	    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
	    table.insert(all_jgates, lst[#lst-1])
            table.insert(predictions,lst[#lst]) 
	    
	    local abscrit = nn.AbsCriterion():cuda()
	    local loss1 = abscrit:forward(all_jgates[t], zerotensors)
	    loss_max = loss_max + loss1
	    if all_jgates[t]:nonzero():dim()>0 then
	        numNZGates = numNZGates + all_jgates[t]:nonzero():size(1)
	    end
	end
	-- find the max scores across all frames for all classes
	local maxscrlyer = nn.CMaxTable():cuda()
	local clscrit = nn.BCECriterion():cuda()
	local videoPreds = maxscrlyer:forward(predictions)
	loss_cls = clscrit:forward(videoPreds,y)
      end
    -- carry over lstm state
	rnn_state[0] = rnn_state[#rnn_state]	--?? do I need this ???
	-- compute percentage of nonzeros
	numNZ[(ptr-1)/opt.batch_size+1] = numNZGates/opt.rho/opt.batch_size
	numNZGates = 0
    end
    loss = loss_max * opt.l1_weight + loss_cls
--    loss = loss/data.trainFeats:size(1)
    local sparsity = 1-numNZ:mean()
    return loss, all_jgates,loss_max, loss_cls, sparsity
end



function feval(x)
    if x~=params then params:copy(x) end
    grad_params:zero()
    ------------ get mini batch ------------
    local x,y = loader:next_batch(1)	-- modify batch_loader.lua
    x,y = prepro(x,y)		-- size of y: rho*batch_size*nClasses

    local rnn_state = {[0] = init_state_global}
    local predictions = {}	-- softmax outputs? 
    local all_jgates = {}	-- j_gates from all LSTMs
    local para_crits = {}
    local loss = 0

    local zerotensors = torch.Tensor(opt.batch_size):zero():cuda() -- ???
    
--[[    for t=1,opt.rho do
	clones.rnn[t]:training()
	------------------ forward pass ----------------
	local prev_c, prev_h, j_gate = unpack(rnn_state[t-1])
	local lst = clones.rnn[t]:forward({x[t],prev_c,prev_h})
	rnn_state[t] = {}
-- init_state={c,h,j_gate}, and lst={c,h,j_gate,pred}, where pred:nClass
	for i=1,#init_state do table.insert(rnn_state[t],lst[i]) end
	table.insert(all_jgates, lst[#lst-1])
	table.insert(predictions,lst[#lst])
	
	local abscrit = nn.AbsCriterion():cuda()
	local frame_wt = 1
-- ??? Here do we need different weights for different frames in time sequence?
-- e.g. frame_wt = t/opt.rho so the later frames weigh more than the beginning
	--local clscrit = nn.ClassNLLCriterion():cuda()
	local clscrit = nn.BCECriterion():cuda()
	local para_criterion = nn.ParallelCriterion():cuda()
	para_criterion:add(abscrit,opt.l1_weight):add(clscrit,frame_wt)
	table.insert(para_crits, para_criterion)
	--y:rho*batch_size*nClasses
	loss = loss + para_crits[t]:forward({all_jgates[t],predictions[t]},{zerotensors,y[t]})
    end
    loss = loss/opt.rho

    local drnn_state = {[opt.rho] = clone_list(init_state, true)} -- true zeros the clones
    for t=opt.rho,1,-1 do
	------------------ backward pass ----------------
	dlstout = para_crits[t]:backward({all_jgates[t],predictions[t]},{zerotensors,y[t]})
	-- dlsout={djGate, doutput}, max backprop to compute d_jgate and lstm backprop.
	table.insert(drnn_state[t],dlstout[2])
	-- backprop for lstm:
	local prev_c,prev_h,j_gate = unpack(rnn_state[t-1])
	local dlst = clones.rnn[t]:backward({x[t],prev_c,prev_h},drnn_state[t])
	-- dlst = {dx, dc, dh}
	drnn_state[t-1] = {}	-- {dc,dh,dj_gate}
	for k,v in pairs(dlst) do 	-- k==1, gradient on x, disgard
	    if k>1 then drnn_state[t-1][k-1] = v end
	end
	table.insert(drnn_state[t-1], dlstout[1])
    end
    ----------------- transfer final state to initial state --------------
    init_state_global = rnn_state[#rnn_state]
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
]]--
  if opt.task == 'localization' then 
    for t=1,opt.rho do
	clones.rnn[t]:training()
	------------------ forward pass ----------------
	local prev_c, prev_h = unpack(rnn_state[t-1])
	local lst = clones.rnn[t]:forward({x[t],prev_c,prev_h})
	rnn_state[t] = {}
-- init_state={c,h}, and lst={c,h,j_gate,pred}, where pred:nClass
	for i=1,#init_state do table.insert(rnn_state[t],lst[i]) end
	table.insert(all_jgates, lst[#lst-1])
	table.insert(predictions,lst[#lst])
	
	local abscrit = nn.AbsCriterion():cuda()
	local frame_wt = 1
-- ??? Here do we need different weights for different frames in time sequence?
-- e.g. frame_wt = t/opt.rho so the later frames weigh more than the beginning
	local clscrit = nn.BCECriterion():cuda()
	local para_criterion = nn.ParallelCriterion():cuda()
	para_criterion:add(abscrit,opt.l1_weight):add(clscrit,frame_wt)
	table.insert(para_crits, para_criterion)	-- para_criterion also learns the weight
	--y:rho*batch_size*nClasses
	loss = loss + para_crits[t]:forward({all_jgates[t],predictions[t]},{zerotensors,y[t]})
    end
    loss = loss/opt.rho

    local drnn_state = {[opt.rho] = clone_list(init_state, true)} -- true zeros the clones
    for t=opt.rho,1,-1 do
	------------------ backward pass ----------------
	local dlstout = para_crits[t]:backward({all_jgates[t],predictions[t]},{zerotensors,y[t]})
	-- dlsout={djGate, doutput}
	-- drnn_state={dc,dh}
	local prev_c,prev_h= unpack(rnn_state[t-1])
	local dc,dh= unpack(drnn_state[t])
	local djGate,doutput = unpack(dlstout)
	local dlst = clones.rnn[t]:backward({x[t],prev_c,prev_h},{dc,dh,djGate,doutput})	
	-- dlst = {dx, dc, dh}
	drnn_state[t-1] = {}	-- {dc,dh}
	for k,v in pairs(dlst) do 	-- k==1, gradient on x, disgard
	    if k>1 then drnn_state[t-1][k-1] = v end
	end
    end
  elseif opt.task == 'classification' then
    local para_crit = nn.ParallelCriterion():cuda()
    local abscrit = nn.AbsCriterion():cuda()
    local clscrit = nn.BCECriterion():cuda()
    local zeroTable = {}
------------------ forward pass ----------------
    for t=1,opt.rho do
	clones.rnn[t]:training()
	local prev_c, prev_h = unpack(rnn_state[t-1])
	local lst = clones.rnn[t]:forward({x[t],prev_c,prev_h})
	rnn_state[t] = {}
-- init_state={c,h}, and lst={c,h,j_gate,pred}, where pred:nClass
	for i=1,#init_state do table.insert(rnn_state[t],lst[i]) end
	table.insert(all_jgates, lst[#lst-1])
	table.insert(predictions,lst[#lst])
	table.insert(zeroTable,zerotensors)
	para_crit:add(abscrit,opt.l1_weight)
    end
    -- find the max scores across all frames for all classes
    local maxscrlyer = nn.CMaxTable():cuda()
    local videoPreds = maxscrlyer:forward(predictions)
    para_crit:add(clscrit,1)
    local paraInputs = all_jgates
    table.insert(paraInputs,videoPreds)
--print(paraInputs)
--print(all_jgates)
    local paraOutputs = zeroTable
    table.insert(paraOutputs,y)
--print(paraOutputs)
-- paraInputs={all_jgates,videoPreds};
-- paraOutputs={zeroTable,y}
    loss = para_crit:forward(paraInputs, paraOutputs)
    -- loss = para_crit:forward({unpack(all_jgates),videoPreds},{unpack(zeroTable),y})
    loss = loss/opt.rho
------------------ backward pass ----------------
    local drnn_state = {[opt.rho] = clone_list(init_state, true)} -- true zeros the clones
    local dlstout = para_crit:backward(paraInputs,paraOutputs)
    -- local dlstout = para_crit:backward({unpack(all_jgates),videoPreds},{unpack(zeroTable),y})
    -- dlsout={dall_jates, doutputMax}, dlstout[t]: dall_jgates[t]
    local dpreds = maxscrlyer:backward(predictions,dlstout[#dlstout])
    -- dpreds={dpreds[1],...,dpreds[rho]}
    for t=opt.rho,1,-1 do	
	local prev_c,prev_h = unpack(rnn_state[t-1])
	local dc,dh = unpack(drnn_state[t])
 	local dparaOutputs = {}
	table.insert(dparaOutputs,dc)
	table.insert(dparaOutputs,dh)
	table.insert(dparaOutputs,dlstout[t])
	table.insert(dparaOutputs,dpreds[t])
	local dlst = clones.rnn[t]:backward({x[t],prev_c,prev_h},dparaOutputs)
	-- local dlst = clones.rnn[t]:backward({x[t],prev_c,prev_h},{unpack(drnn_state[t]),dlstout[t],dpreds[t]})	
	-- dlst = {dx, dc, dh}
	drnn_state[t-1] = {}	-- {dc,dh}
	for k,v in pairs(dlst) do 	-- k==1, gradient on x, disgard
	    if k>1 then drnn_state[t-1][k-1] = v end
	end
    end 
  end
----------------- transfer final state to initial state --------------
    init_state_global = rnn_state[#rnn_state]
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end


-- ============================ Training ================================
function trainer:train()
    epoch = 1
-- global variables:
    init_state_global = clone_list(init_state)
    train_losses = {}
    val_losses = {}
    losses_max = {}
    losses_cls = {}

    local num_batches = torch.floor(data.trainFeats:size(1)/opt.batch_size)
    local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
--print(optim_state)
    local iterations = opt.max_epochs * num_batches
    local iterations_per_epoch = num_batches
    local loss0 = nil

    print(sys.COLORS.green .. 'Training begins...')
    local timer_total = torch.Timer()
    for i = 1, iterations do 
	epoch = i/num_batches
	local timer = torch.Timer()
	local _, loss = optim.adam(feval, params, optim_state)
	local time = timer:time().real/60 
	local train_loss = loss[1]
	train_losses[i] = train_loss

	if epoch>=opt.learning_rate_decay_after then
	    local decay_factor = opt.learning_rate_decay
--print(optim_state)
	    optim_state.learningRate = optim_state.learningRate * decay_factor
	end

	if i % opt.eval_val_every == 1 or i == iterations then
	    val_loss, all_gates, loss_max, loss_cls, j_sp = eval_split(2)
	    table.insert(val_losses, val_loss)
	    table.insert(losses_max, loss_max)
	    table.insert(losses_cls, loss_cls)	
print(loss_max)
print(loss_cls)
print(val_loss)
print('Percentage of frames skipped:')
print(j_sp)	-- print out sparsity of j_gate
	end
	if opt.savecheckpoint == true and i==iterations then 
	    paths.mkdir(opt.checkpoint_dir)
	    local checkpointfile = string.format('%s/%s_%s_lambda%f_thrsh%f_sp%f.t7', opt.checkpoint_dir, opt.checkpointfn, opt.task, opt.l1_weight, opt.thrsh, j_sp)
	    print('saving checkpoint to '..checkpointfile)
	    print(sys.COLORS.green .. ' -------> evaluations loss = ' .. val_loss)
	    checkpoint = {}
	    checkpoint.protos = protos
	    checkpoint.opt = opt
	    checkpoint.train_losses = train_losses
   	    checkpoint.val_loss = val_loss
	    checkpoint.val_losses = val_losses
   	    checkpoint.epoch = epoch
   	    torch.save(checkpointfile, checkpoint)
	end

	if i % opt.print_every == 1 then
	    print(string.format('%d/%d (epoch %.2f), train_loss = %6.8f, time/batch = %.4fs', i, iterations, epoch, train_loss, time))
	end

	if i % 10 == 0 then collectgarbage() end
	if loss0 == nil then loss0 = loss[1] end
	if loss[1]~=loss[1] then
	    print('loss is NaN. This usually indicates a bug...')
	    break
	end
    end
    print(string.format('Total training time: %f minutes',timer_total:time().real/60))
    self.protos = protos
    return self
end

return trainer

