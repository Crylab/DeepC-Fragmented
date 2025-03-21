clear all
clc
disp('Run multiple realisations of disturbance and sensor noise')
disp('Run each realisation with varying prediction horizon for both segmented and unsegmented formulations:')
% Run for the following prediction horizons:
Nlist = [10]%,20,40]; % List of horizons for which to run

N = 10;
simlength_tr = 50;
number_seeds = 200;
Slist = 1:number_seeds; % Seeds to use for different random component realisations

%simlength_tr = 60; % number of training samples
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
yspfull = 0.8*[0.5*ones(spch1,1);-0.3*ones(spch2,1);-0.1*ones(spch1,1);...
    -0.5*ones(spch2,1);0.5*ones(spch1,1);-0.1*ones(spch2,1);0.5*ones(spch1,1);...
    -0.5*ones(runlen-spch1*4-spch2*3+max(Nlist)*dtc,1)];
sp_band = 0; % Set point band

% Build an input cost
cost = ones(runlen/dtc+max(Nlist),1);

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
seedcount = 0;
%%

for segmented = [0, 1] 
    error_list = zeros(1, number_seeds);   
    number_of_fails = 0;
    for rseed = Slist % Run for different realisations of random components
        rng(rseed)
        if segmented == 1
            disp("Run Segmented with Seed="+num2str(rseed))
        else
            disp("Run Original with Seed="+num2str(rseed))
        end
        % Build a random input for training
        urand=max(umin*1,min(umax*1,4*(-0.5+rand(ceil(trlen/(trgap)),1))));
        % Build a random disturbance component
        wrandtr = 0.2*(1+sin((1/15)*(1:trlen)))'+0.3*(-0.5+rand(trlen,1));
        wrand = 0.2*(1+sin((1/15)*(1:runlen/dtc)))'+0.3*(-0.5+rand(runlen/dtc,1));
        % Build a random sensor noise
        sensornoise = 0.1*(randn(trlen+runlen+1,1));
    
        includedist = 1; % Run with and without external disturbance
        sensornoise = (includedist)*sensornoise;
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

        n_ord = 10;
        %Ctrlparams.T = (1+1)*(Ctrlparams.Tf+Ctrlparams.Tini+1*n_ord)-1;
        Ctrlparams.T = simlength_tr - (Ctrlparams.Tf+Ctrlparams.Tini);



        % Objective penalty weight (\lambda_g)
        Ctrlparams.lamg = 0.5;    


        %% 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Train  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Run simulation model for training period
        disturb = includedist*wrandtr;
        [ttr,ytr,u_tr,ws] = another_mass(linspace(1,trlen+1,ceil(trlen/dtc)+1),y0,1,m1,m2,k1,k2,d1,d2,[],urand,disturb,trlen,trgap);

        % Gather data
        Meas.uin = u_tr(1:end);
        Meas.y = ytr(1:end,2)+sensornoise(1:trlen+1);

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
            usp = CtrlSeg(Data.uini,Data.yini,Fcasts,umax,umin,Ctrlparams);
        else
            usp = CtrlUnseg(Data.uini,Data.yini,Fcasts,umax,umin,Ctrlparams);
        end

        t = ttr;
        yr = ytr(end,:);

        % Pre allocation
        trun = zeros(runlen/dtc,1); % Time vector
        yrun = zeros(runlen/dtc,4); % Output vector
        wrun = zeros(runlen/dtc,1); % Disturbance vector
        urun = zeros(runlen/dtc,1); % Input vector

        no_error = 0;


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
            Meas.y = [ytr(1:end,2)+sensornoise(1:trlen+1);yrun(1:k,2)+sensornoise(trlen+2:trlen+k+1)];

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
                no_error = 100000;
                break
            end
            % Collect computational time measurement
            timey(k,runcount) = ti;
        end

        % Measure and store set-point error and computational time
        if no_error > 0
            number_of_fails = number_of_fails +1;
            continue
        else
            trajectories(:, rseed - number_of_fails + (segmented * number_seeds)) = yrun(:,2);
        end               

    end
    disp("Algorithm failed: "+num2str(number_of_fails))
end
save("trajectories", "trajectories")
