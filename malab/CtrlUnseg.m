function [u,y_sp,ufull,y_sp_full,ti] = CtrlUnseg(uini,yini,Fcasts,umax,umin,Ctrlparams)
%%%%%%%%%%%%%%%%%%%%%%%
% Define params
%%%%%%%%%%%%%%%%%%%%%%%
T = Ctrlparams.T;
Tini = Ctrlparams.Tini;
Tf = Ctrlparams.Tf;
lamg = Ctrlparams.lamg;
Up=Ctrlparams.Up;
Uf=Ctrlparams.Uf;
Yp=Ctrlparams.Yp;
Yf=Ctrlparams.Yf;

ysp = Fcasts.ysp_lo;
ysp_up = Fcasts.ysp_hi;

num_g = (T-Tini-Tf+1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Some useful intermediates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PI = pinv([Up;Yp;Uf])*[Up;Yp;Uf];
PHI=(eye(num_g)-PI)'*(eye(num_g)-PI);
PHI=(PHI+PHI')/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constraints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Equality:
% Up*g = uini
Aeq=[Up,zeros(Tini),zeros(Tini,Tf)];
beq=uini;

% Inequality:
% umin <= Uf*g <= umax
a2=[-Uf,zeros(Tf,Tini),zeros(Tf)];
b2=-umin*ones(Tf,1);
a3=[Uf,zeros(Tf,Tini),zeros(Tf)];
b3=umax*ones(Tf,1);
% epsilon_y >= |Yf*g - setpoint|
a4=[Yf,zeros(Tf,Tini),-eye(Tf)];
b4=ysp_up;
a5=[-Yf,zeros(Tf,Tini),-eye(Tf)];
b5=-ysp;
% epsilon_y >= 0
a6=[zeros(Tf,num_g),zeros(Tf,Tini),-eye(Tf)];
b6=zeros(Tf,1);
% epsilon_ini >= |Yp*g - yini|
a7=[Yp,-eye(Tini),zeros(Tini,Tf)];
b7=yini;
a8=[-Yp,-eye(Tini),zeros(Tini,Tf)];
b8=-yini;
% epsilon_ini >= 0
a9=[zeros(Tini,num_g),-eye(Tini),zeros(Tini,Tf)];
b9=zeros(Tini,1);

% Gather inequality constraints
Aineq=[a2;a3;a4;a5;a6;a7;a8;a9];
bineq=[b2;b3;b4;b5;b6;b7;b8;b9];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Objectives and solvers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% First objective: minimize sum(epsilon_ini)
f=[zeros(1,num_g),ones(1,Tini),zeros(1,Tf)];
optionsL = optimoptions('linprog','Display','none');
gopt0=linprog(f,Aineq,bineq,Aeq,beq,[],[],optionsL);
acc=max(0,f*gopt0);

% Second objective: minimize ||(I-PI)g||^2_2 + sum(epsilon_y)
f1=[zeros(1,num_g),zeros(1,Tini),ones(1,Tf)];
H=[lamg*PHI,zeros(num_g,Tini),zeros(num_g,Tf);zeros(Tini+Tf,num_g+Tini+Tf)];
optionsQ = optimoptions('quadprog','LinearSolver','dense','Display','off');
tStart=tic;
gopt=quadprog(H,f1,[Aineq;f],[bineq;acc],Aeq,beq,[],[],[],optionsQ);

% Timer
ti=toc(tStart);

% Take first output of solution
y_sp_full = Yf(1:Tf,:)*gopt(1:num_g);
y_sp = y_sp_full(1);

ufull = Uf(1:Tf,:)*gopt(1:num_g);
u = ufull(1);
end