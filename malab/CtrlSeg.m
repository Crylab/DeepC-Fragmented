function [u,y_sp,ufull,y_sp_full,ti] = CtrlSeg(uini,yini,Fcasts,umax,umin,Ctrlparams)
%%%%%%%%%%%%%%%%%%%%%%%
% Define params
%%%%%%%%%%%%%%%%%%%%%%%
T = Ctrlparams.T;
Tini = Ctrlparams.Tini;
Tf = Ctrlparams.Tf;
lamg = Ctrlparams.lamg;
Up = Ctrlparams.Up;
Uf = Ctrlparams.Uf;
Yp = Ctrlparams.Yp;
Yf = Ctrlparams.Yf;

ysp = Fcasts.ysp_lo;
ysp_up = Fcasts.ysp_hi;
cost = Fcasts.cost;

num_g = (T-Tini-Tf+1);
folds = ceil(length(cost)/Tf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct setpoint vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%
spvect = zeros(numel(ysp),1);
spvect_up = zeros(numel(ysp),1);
for ii=1:folds
    spvect((ii-1)*Tf+1:ii*Tf) = reshape(ysp((ii-1)*Tf+1:ii*Tf,:),Tf,1);
    spvect_up((ii-1)*Tf+1:ii*Tf) = reshape(ysp_up((ii-1)*Tf+1:ii*Tf,:),Tf,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some useful intermediates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
UpUf = kron(diag(ones(folds-1,1),-1),-Uf)+kron(eye(folds),Up);
YpYf = kron(diag(ones(folds-1,1),-1),-Yf)+kron(eye(folds),Yp);

PI = pinv([Up;Yp;Uf])*[Up;Yp;Uf];
PHI = (eye(num_g*folds)-kron(eye(folds),PI))'*(eye(num_g*folds)-kron(eye(folds),PI));
PHI = (PHI+PHI')/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constraints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Equality:
% Up*g = uini
Aeq=[UpUf,zeros(Tini*folds,(2*Tini)*folds)];
beq=[uini;zeros(Tini*(folds-1),1)];

% Inequality:
% umin <= Uf*g <= umax
a2=[kron(eye(folds),-Uf),zeros(Tini*folds,(2*Tini)*folds)];
b2=-umin*ones(folds*Tini,1);
a3=[kron(eye(folds),Uf),zeros(Tini*folds,(2*Tini)*folds)];
b3=umax*ones(folds*Tf,1);
% epsilon_ini >= 0
a4=[zeros(Tini*folds,num_g*folds),-eye(Tini*folds),zeros(Tini*folds)];
b4=zeros(Tini*folds,1);
% epsilon_ini >= |Yp*g - yini|
a5=[-YpYf,-eye(Tini*folds),zeros(Tini*folds)];
b5=[-yini;zeros(Tini*(folds-1),1)];
a6=[YpYf,-eye(Tini*folds),zeros(Tini*folds)];
b6=[yini;zeros(Tini*(folds-1),1)];
% epsilon_y >= |Yf*g - setpoint|
a7=[kron(eye(folds),Yf),zeros(Tini*folds),-eye(Tini*folds)];
b7=spvect_up;
a8=[-kron(eye(folds),Yf),zeros(Tini*folds),-eye(Tini*folds)];
b8=-spvect;
% epsilon_y >= 0
a9=[zeros(Tini*folds,num_g*folds),zeros(Tini*folds),-eye(Tini*folds)];
b9=zeros(Tini*folds,1);

% Gather inequality constraints
Aineq=[a2;a3;a4;a5;a6;a7;a8;a9];
bineq=[b2;b3;b4;b5;b6;b7;b8;b9];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Objectives and solvers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% First objective: minimize sum(epsilon_ini)
f=[zeros(1,num_g*folds),ones(1,Tini*folds),zeros(1,Tini*folds)];
optionsL = optimoptions('linprog','Display','none');
gopt0=linprog(f,Aineq,bineq,Aeq,beq,[],[],optionsL);
acc=max(0,f*gopt0);

% Second objective: minimize ||(I-PI)g||^2_2 + sum(epsilon_y)
f1=[zeros(1,num_g*folds),zeros(1,Tini*folds),ones(1,Tini*folds)];
H=[lamg*PHI,zeros(num_g*folds,Tini*folds),zeros(num_g*folds,Tini*folds);zeros(2*folds*Tini,folds*(num_g+Tini*2))];
optionsQ = optimoptions('quadprog','LinearSolver','sparse','Display','off');
tStart=tic;
gopt=quadprog(sparse(H),f1,[Aineq;f],[bineq;acc],Aeq,beq,[],[],[],optionsQ);

% Timer
ti=toc(tStart);

% Take first output of solution
y_sp_full = kron(eye(folds),Yf(1:Tf,:))*gopt(1:num_g*folds);
y_sp = y_sp_full(1);

ufull = kron(eye(folds),Uf(1:Tf,:))*gopt(1:num_g*folds);
u = ufull(1);

end