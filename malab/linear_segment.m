disp('Run scenarios with varying prediction horizon:')


% Sim setup
simlength_tr = 300; % number of training samples
simlength_run = 100; % number of run samples

% Model physical params
m1 = 0.5;
m2 = 1.5;
k1 = 2;
k2 = 2;
d1 = 1;
d2 = 1;
y0 = [0;0;0;0]; % initial condition

% Max/min input limits
umax = 1;
umin = -1;


dtc = 1; % control sample time
trlen = simlength_tr*dtc; % training length in seconds
runlen = simlength_run*dtc; % run length in seconds


% Build a setpoint to follow:
spch1 = 25;
spch2 = 20;
yspfull = 0.8*[0.5*ones(spch1,1);-0.3*ones(spch2,1);-0.1*ones(spch1,1);-0.5*ones(spch2,1);0.5*ones(spch1,1);-0.1*ones(spch2,1);0.5*ones(spch1,1);-0.5*ones(runlen-spch1*4-spch2*3+100*dtc,1)];
sp_band = 0; % Set point band


% Important things are here
n_seeds = 100;
disp("Number of seeds = "+num2str(n_seeds))
segmented = 1; % run is segmented
for prediction_horizion = [10, 20, 40]
    for segmented = 0:1
    
        if segmented == 1
            disp("Segmented DeePC with Prediction = "+num2str(prediction_horizion))
        else
            disp("Original DeePC with Prediction = "+num2str(prediction_horizion))
        end
        
        errfad = zeros(n_seeds, 1); % for collecting set-point errors
        
        for seed = 1:n_seeds
        
            % Build a random input for training
            rng(seed);
            trgap = 10;
            urand=max(umin*1,min(umax*1,4*(-0.5+rand(ceil(trlen/(trgap)),1))));
        
            % Build a random disturbance component
            wrandtr = 1*(0.2*(1+sin((1/15)*(1:trlen)))'+0.3*(-0.5+rand(trlen,1)));
            wrand = 1*(0.2*(1+sin((1/15)*(1:runlen/dtc)))'+0.3*(-0.5+rand(runlen/dtc,1)));
        
            % Build an input cost
            cost = ones(runlen/dtc+100,1);
        
            % Preallocation
            timey = zeros(runlen/dtc,1); % for collecting computational times
            
            timeyfad = zeros(4,1); % for collecting average computational times
        
        
            includedist = 1;
        
            scenariocount = 1;
        
            
        
            runcount = 1; % counter
        
        
            N = prediction_horizion;
                
            % Controller params
            Ctrlparams.Tini = 5;
            Ctrlparams.Tf = N;
        
            if segmented == 1
                Horiz = floor(Ctrlparams.Tf/Ctrlparams.Tini)*Ctrlparams.Tini;
                Ctrlparams.Tf = Ctrlparams.Tini;
            else
                Horiz = Ctrlparams.Tf;
            end
        
            n_ord = 10;
            Ctrlparams.T = 2*(Ctrlparams.Tf+Ctrlparams.Tini+n_ord)-1;
        
        
        
            % Objective penalty weight (\lambda_g)
            Ctrlparams.lamg = 0.5;    
        
        
            %% 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Train  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
            % Run simulation model for training period
            disturb = includedist*wrandtr;
            [ttr,ytr,u_tr,ws] = another_mass(linspace(1,trlen+1,ceil(trlen/dtc)+1),y0,1,m1,m2,k1,k2,d1,d2,[],urand,disturb,trlen,trgap);
        
            % Gather data
            Meas.uin = u_tr(1:end);
            Meas.y = ytr(1:end,2);
        
            % Build data matrix structures
            [Data] = BuildDataMatrices(trlen/dtc,Meas,Ctrlparams);
            [Ctrlparams.Up,Ctrlparams.Uf,Ctrlparams.Yp,Ctrlparams.Yf] = getHankels(Data.utr,Data.ytr,Ctrlparams);
        
            % Get set-point forecasts
            Fcasts.ysp_lo = yspfull(1:Horiz);
            Fcasts.ysp_hi = yspfull(1:Horiz)+sp_band;
            Fcasts.cost = cost(1:Horiz);
        
        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Run  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
            % Call data predictive control algorithm for 1st time to get first input
            if segmented==1
                usp = CtrlSegment(Data.uini,Data.yini,Fcasts,umax,umin,u_tr(end),Ctrlparams);
            else
                usp = CtrlUnsegment(Data.uini,Data.yini,Fcasts,umax,umin,Ctrlparams);
            end
        
            t = ttr;
            yr = ytr(end,:);
        
            % Pre allocation
            trun = zeros(runlen/dtc,1); % Time vector
            yrun = zeros(runlen/dtc,4); % Output vector
            wrun = zeros(runlen/dtc,1); % Disturbance vector
            urun = zeros(runlen/dtc,1); % Input vector
        
        
            for k = 1:runlen/dtc
        
                % Update setpoints
                Fcasts.ysp_lo = yspfull(k+1:k+Horiz);
                Fcasts.ysp_hi = yspfull(k+1:k+Horiz)+sp_band;
                Fcasts.cost = cost(k+1:k+Horiz);
                w = includedist*wrand(k);
        
                % Call simulation model for 1 timestep
                [t,yr,u_r,wsr] = another_mass([t(end),t(end)+dtc],yr(end,:),0,m1,m2,k1,k2,d1,d2,usp,urand,w,trlen,trgap);
                
                yrun(k,:) = yr(end,:);
                trun(k) = t(end);
                urun(k) = u_r(end);
                wrun(k) = wsr(end);
                
                % Gather data
                Meas.uin = [u_tr(1:end);urun(1:k)];
                Meas.y = [ytr(1:end,2);yrun(1:k,2)];
                
                % Update initialisation matrices
                [Data] = BuildDataMatrices(round(t(end)/dtc),Meas,Ctrlparams);
                        
        
                % Call data predictive control algorithm
                if segmented==1
                    [usp,~,~,~,ti] = CtrlSegment(Data.uini,Data.yini,Fcasts,umax,umin,u_r(end),Ctrlparams);
                else
                    [usp,~,~,~,ti] = CtrlUnsegment(Data.uini,Data.yini,Fcasts,umax,umin,Ctrlparams);
                end
        
                % Collect computational time measurement
                timey(k,runcount) = ti;
                
            end
        
            % Measure and store set-point error and computational time
            err = abs(yrun(:,2)-yspfull(1:runlen/dtc));
            errfad(seed)=sum(err);
        end 
        disp("Mean value: "+num2str(mean(errfad)))
        disp("Std Dev: "+num2str(std(errfad)))
        disp("Max val: "+num2str(max(errfad)))
        disp("Min val: "+num2str(min(errfad)))
    end
end
