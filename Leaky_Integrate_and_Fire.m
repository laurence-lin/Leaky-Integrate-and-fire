%% Test LIF model for pattern recognition on Iris datset
clc; clear all; clf;

%% data set
data = importdata('Iris_train.txt');
seq = randperm(length(data)); %% random sequence from 1 to length of data
data = data(seq,:); %% random sort the data
feature = data(:,1:4);
target = data(:,5);
%% take 120 for train data, 30 for test data
train = feature(1:120,:);
train_target = target(1:120,1);
test = feature(121:150,:);
test_target = target(121:150,1);
len = length(train);

%% indices of each class
c1 = find(train_target==1); 
c2 = find(train_target==2);
c3 = find(train_target==3);

%% normalization
for i = 1:2
    train(:,i) = (train(:,i)-mean(train(:,i)))/(max(train(:,i))-min(train(:,i))); 
    test(:,i) = (test(:,i)-mean(test(:,i)))/(max(test(:,i))-min(test(:,i))); 
end

%% initial LIF parameter
threshold = 50; 
restPotential = -60;
a = 0.5;
b = -0.001;
c = -50;
gain = 0.1;
weight = rand(4,1);
I = train*weight*gain; %% fixed encoded InputCurrent
T = 1000; %% firing interval 

PSP = zeros(1,T);
actionPotential = zeros(1,T); %% spike train for a input current
PSP(1,1) = restPotential;

lastPSP = restPotential;
lastAction = 0;

fr = zeros(len,1); 

for num = 1:len
    for time = 1:T
        PSP(1,time) = lastPSP + I(num) + a - b*lastPSP; %% accumulate current PSP with input current   
        if PSP(1,time) > threshold
            lastPSP = c;
            actionPotential(1,time) = 1;
        elseif PSP(1,time) < threshold
            lastPSP = PSP(1,time);
            actionPotential(1,time) = 0;   
        end
        lastAction = actionPotential(1,time);
    end
    spikes = find(actionPotential==1);
    spike_num = length(spikes);
    firing_rate = spike_num/T;
    fr(num) = firing_rate;  %% firing rate array for training data
    
end

%% average firing rate of each class
AFR1 = mean(fr(c1));
AFR2 = mean(fr(c2));
AFR3 = mean(fr(c3));
%% standard deviation of firing rate of each class
SDFR1 = sqrt(sum((fr(c1)-AFR1).^2)/length(c1));
SDFR2 = sqrt(sum((fr(c2)-AFR2).^2)/length(c2));
SDFR3 = sqrt(sum((fr(c3)-AFR3).^2)/length(c3));

%% train synapse weight
%% differential evolution algorithm
%% DE pamameter
MAXGEN = 1000; %% max generation; the maximum number of childs generated
F = 0.8; %%control vector
XMAX = 1; %% max value of these parameters
XMIN = 0; %% min value of these parameters
CR = 0.9; %% hybrid control parameter
NP = 40; %% populations per cycle(generate NP solutions at first)

% initialize population for first generation
gene = rand(NP,4);
best = zeros(1,4);

for iteration = 1:(MAXGEN/NP)
    for i = 1:NP
        target_vector = gene(i,:); %% initial target vector is the first array
        index = randperm(NP);
        index(find(index==i)) = []; % r1!=r2!=r3!=i
        r = index(1:3); % r = [r1,r2,r3]
        % mutation (the mutation output should be constrainted in [XMIN,XMAX])
        h = gene(r(1),:) + F*(gene(r(2),:)-gene(r(3),:));
        for k = 1:size(h,2)
            if h(k) > XMAX
                h(k) =  XMIN + (XMAX-XMIN)*rand;
            elseif h(k) < XMIN
                h(k) =  XMIN + (XMAX-XMIN)*rand;
            end
        end    
        % crossover
        v = zeros(size(h));
        for j = 1:size(v,2)
            param = XMIN + (XMAX-XMIN)*rand;
            if param <= CR
                v(j) = h(j);
            else
                v(j) = target_vector(j);
            end
        end 
        
        %selection
        if fitness(train,c1,c2,c3,v) < fitness(train,c1,c2,c3,target_vector) %if new weight obtain a better fitness function 
            gene(i,:) = v; %% update new generation
        else
            gene(i,:) = target_vector;
        end
        if fitness(train,c1,c2,c3,gene(i,:)) < fitness(train,c1,c2,c3,best) %% check if the current solution is better, update best solution
            best = gene(i,:);
        end
        fitness(train,c1,c2,c3,best)
    end
    
end
figure(1);
hold on
weight = best
plot(test_target,'bo')
%% test

I = test*weight'*gain;
class = zeros(size(test_target));

PSP = zeros(1,T);
actionPotential = zeros(1,T); %% spike train for a input current
PSP(1,1) = restPotential;

lastPSP = restPotential;
lastAction = 0;

fr = zeros(len,1); 
%% test data
for num = 1:length(test)
    for time = 1:T
        PSP(1,time) = lastPSP + I(num) + a - b*lastPSP; %% accumulate current PSP with input current   
        if PSP(1,time) > threshold
            lastPSP = c;
            actionPotential(1,time) = 1;
        elseif PSP(1,time) < threshold
            lastPSP = PSP(1,time);
            actionPotential(1,time) = 0;   
        end
        lastAction = actionPotential(1,time);
    end
    spikes = find(actionPotential==1);
    spike_num = length(spikes);
    firing_rate = spike_num/T;
    fr(num) = firing_rate;  %% Elements in I corresponds to fr
    d = zeros(1,3);
    d(1) = abs(AFR1-fr(num));
    d(2) = abs(AFR2-fr(num));
    d(3) = abs(AFR3-fr(num));
    class(num) = find(d == min(d));
end

count = 0;
for i = 1:length(test)
    if test_target(i) == class(i)
        count = count + 1;
    end
end
error = 1 - count/length(test);

plot(class,'rx')
title(['Error rate=',num2str(error)]);
figure(5);
plot(PSP,'b');
hold off

