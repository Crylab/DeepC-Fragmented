clear all;
clc;
disp('Run multiple realisations pendulum tracking with varying datasetsize')
disp('Run each realisation with varying prediction horizon for both segmented and unsegmented formulations:')

% Run for the following prediction horizons:
N = 10; % Prediction horizons for which to run
Nlist = [N];
Slist = 1:100; % Seeds to use for different random component realisations
% Sim setup
dataset_list = [30, 50, 100, 200];
damping_list = [0.25];
for dataset_index = 1:length(dataset_list)
    simlength_tr = dataset_list(dataset_index);
    %simlength_tr = 300; % number of training samples
    simlength_run = 100; % number of run samples

    y0 = [0;0;0;0]; % initial condition

    % Max/min input limits
    umax = 20;
    umin = -20;

    dtc = 0.2; % control sample time
    trlen = simlength_tr*dtc; % training length in seconds
    runlen = simlength_run*dtc; % run length in seconds


    % Build a setpoint to follow:
    spch1 = 25;
    spch2 = 20;
    yspfull = 0.8*[0.0*ones(5,1);0.5*ones(spch1,1);-0.3*ones(spch2,1);-0.1*ones(spch1,1);...
        -0.5*ones(spch2,1);0.5*ones(spch1,1);-0.1*ones(spch2,1);0.5*ones(spch1,1);...
        -0.5*ones(runlen-spch1*4-spch2*3+max(Nlist)*dtc,1)];

    yspfull = yspfull * pi;

    sp_band = 0; % Set point band

    % Build an input cost
    cost = ones(runlen/dtc+max(Nlist)+5,1);

    % Preallocation
    timey = zeros(runlen/dtc,length(Nlist)); % for collecting computational times
    errfad = zeros(4,length(Nlist)); % for collecting set-point errors
    timeyfad = zeros(4,length(Nlist)); % for collecting average computational times
    errUnseg = zeros(length(Slist),length(Nlist));
    errSeg = zeros(length(Slist),length(Nlist));
    errUnsegSens = zeros(length(Slist),length(Nlist));
    errSegSens = zeros(length(Slist),length(Nlist));
    timeyUnseg = zeros(length(Slist),length(Nlist));
    timeySeg = zeros(length(Slist),length(Nlist));

    trgap = 10;
    %%
    
    for damping_index = 1:length(damping_list)
        damping = damping_list(damping_index);
        for segmented = [0, 1] % First run is unsegmeneted, second run is segmented
            %error_list = zeros(1, 10);
            if segmented == 1
                disp("Run Segmented with Dataset="+num2str(simlength_tr))
            else
                disp("Run Original with Dataset="+num2str(simlength_tr))
                if simlength_tr < 50
                    disp("Dataset is too small for original DeePC")
                    continue
                end
            end
            for rseed = Slist % Run for different realisations of random components
                %damping = (100+rseed)/1000;
                rng(rseed)
                % Build a random set-points for training
                target = reshape(repmat(-pi + (2*pi)*rand(1, 30), 10, 1), [], 1);
                scenariocount = 1;  
                runcount = 1; % counter
                
                % Controller params
                Ctrlparams.Tini = 5;
                Ctrlparams.Tf = N;
        
                if segmented == 1
                    Horiz = floor(Ctrlparams.Tf/Ctrlparams.Tini)*Ctrlparams.Tini;
                    Ctrlparams.Tf = Ctrlparams.Tini;
                else
                    Horiz = Ctrlparams.Tf;
                end
        
                %n_ord = 50;
                Ctrlparams.T = simlength_tr - (Ctrlparams.Tf+Ctrlparams.Tini+1);
        
                Ctrlparams.lamg = 0.5;    
        
        
                %% 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Train  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
                % Run simulation model for training period
                u_tr = zeros(simlength_tr, 1);
                ytr = y0';
        
                for i = 1:simlength_tr
                    % Call simulation model for 1 timestep
                    torque = optimal_pendulum(ytr(end,:), target(i));
                    [t,y_last] = double_pendulum([(i-1)*dtc,i*dtc],ytr(end,:),torque, damping);
                    ytr(i,:) = y_last(end,:);
                    u_tr(i) = torque;
                end
        
                % Gather data
                Meas.uin = u_tr(1:end);
                Meas.y = ytr(1:end,1);
        
                % Build data matrix structures
                [Data] = BuildDataMatrices(simlength_tr,Meas,Ctrlparams);
                [Ctrlparams.Up,Ctrlparams.Uf,Ctrlparams.Yp,Ctrlparams.Yf] = getHankels(Data.utr,Data.ytr,Ctrlparams);
        
                % Get set-point forecasts
                Fcasts.ysp_lo = yspfull(1:Horiz);
                Fcasts.ysp_hi = yspfull(1:Horiz)+sp_band;
                Fcasts.cost = cost(1:Horiz);
        
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Run  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
                % Call data predictive control algorithm for 1st time to get first input
                Data.uini = zeros(Ctrlparams.Tini, 1);
                Data.yini = zeros(Ctrlparams.Tini, 1);
        
                if segmented==1
                    usp = CtrlSeg(Data.uini,Data.yini,Fcasts,umax,umin,Ctrlparams);
                else
                    usp = CtrlUnseg(Data.uini,Data.yini,Fcasts,umax,umin,Ctrlparams);
                end
        
                t = 0;
                yr = y0';
        
                % Pre allocation
                trun = zeros(simlength_run,1); % Time vector
                yrun = y0';
                urun = zeros(simlength_run,1); % Input vector
                no_error = 0;
        
                for k = Ctrlparams.Tini+1:simlength_run+Ctrlparams.Tini
        
                    % Update setpoints
                    Fcasts.ysp_lo = yspfull(k+1:k+Horiz);
                    Fcasts.ysp_hi = yspfull(k+1:k+Horiz)+sp_band;
                    Fcasts.cost = cost(k+1:k+Horiz);
        
                    % Call simulation model for 1 timestep
                    [t,y_last] = double_pendulum([(k-1)*dtc,k*dtc],yrun(end,:),usp, damping);
        
                    yrun(k,:) = y_last(end,:);
                    trun(k) = t(end);
                    urun(k) = usp;
        
                    % Gather data
                    Meas.uin = urun(1:k);
                    Meas.y = yrun(1:k,1);
        
                    % Update initialisation matrices
                    [Data] = BuildDataMatrices(round(t(end)/dtc),Meas,Ctrlparams);
        
        
                    % Call data predictive control algorithm
                    try
                        if segmented==1
                            [usp,~,~,~,ti] = CtrlSeg(Data.uini,Data.yini,Fcasts,umax,umin,Ctrlparams);
                        else
                            [usp,~,~,~,ti] = CtrlUnseg(Data.uini,Data.yini,Fcasts,umax,umin,Ctrlparams);
                        end
                    catch ME
                        no_error = Inf;
                        break
                    end
            
                    % Collect computational time measurement
                    timey(k,runcount) = ti;
                end
                
                % Measure and store set-point error and computational time
                if no_error > 0
                    error_list(rseed) = no_error;
                else
                    err = abs(yrun(Ctrlparams.Tini+1:end,1)-yspfull(Ctrlparams.Tini+1:Ctrlparams.Tini+runlen/dtc));
                    error_list(rseed) = sum(err);
                end
                %disp(sum(err));
        
                trajectories(dataset_index+(length(dataset_list)*segmented), rseed) = sum(err);
                
        
                %disp(strcat('N=',num2str(N),' run completed'))    
                runcount = runcount+1;
                
                           
                scenariocount = scenariocount+1;
        
            end

            %disp("Mean value: "+num2str(mean(error_list)))
            %disp("Std Dev: "+num2str(std(error_list)))
            %disp("Max val: "+num2str(max(error_list)))
            %disp("Min val: "+num2str(min(error_list)))
        end
    end
end
save("pendulum_dataset", "trajectories")
