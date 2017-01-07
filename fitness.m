function fit = fitness(train,c1,c2,c3,w)

len = length(train);

%% initial LIF parameter
threshold = 50; 
restPotential = -60;
a = 0.5;
b = -0.001;
c = -50;
gain = 0.1;
I = train*w'*gain; %% initial encoded InputCurrent
T = 1000; %% firing interval 

PSP = zeros(1,T);
actionPotential = zeros(1,T);
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
    spike = find(actionPotential==1);
    spike_num = length(spike);
    firing_rate = spike_num/T;
    fr(num) = firing_rate;  %% Elements in I corresponds to fr 
end

%% average firing rate of each class
AFR1 = mean(fr(c1));
AFR2 = mean(fr(c2));
AFR3 = mean(fr(c3));
%% standard deviation of firing rate of each class
SDFR1 = sqrt(sum((fr(c1)-AFR1).^2)/length(c1));
SDFR2 = sqrt(sum((fr(c2)-AFR2).^2)/length(c2));
SDFR3 = sqrt(sum((fr(c3)-AFR3).^2)/length(c3));
dist = abs(AFR1-AFR2) + abs(AFR2-AFR3) + abs(AFR1-AFR3);
fit = 1/(dist) + SDFR1 + SDFR2 + SDFR3;
   
end