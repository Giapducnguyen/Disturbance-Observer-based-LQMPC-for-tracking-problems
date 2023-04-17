% % Double integrator problem - with disturbance observer
% % Code for the Reference Tracking formulation 2 of
% % Module 5 - Introduction to MPC - Kolmanovsky
clear all
close all
clc

%% Continuous-time model
Ac = [0,1;0,0];
Bc = [0;1];
Cc = [1,0];
Dc = 0;

nx = size(Ac,2);    % number of states
nu = size(Bc,2);    % number of inputs
ny = size(Cc,1);    % number of outputs
nr = 1;             % tracking on x1 only
nd = 1;             % number of disturbances

%% Discrete-time model
Ts = 0.1; % sampling period
sysd = c2d(ss(Ac,Bc,Cc,Dc),Ts); % discretization
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;
Dd = sysd.D;

%% Extend state to include disturbance estimate \hat{w} 
% % state z = [x(k); \hat{w}(k)];

Az = [Ad,                       Bd;
      zeros(nd,size(Ad,2)),     eye(nd)];
Bz = [Bd; zeros(nd,size(Bd,2))];
Cz = [Cd, zeros(size(Cd,1),nd)];

nz = size(Az,2);

%% Extend state to include control increments du and reference r
% % state z = [z(k); u(k-1); r(k)];

Az_ext = [Az,                       Bz,             zeros(size(Az,1),1);
          zeros(nu,size(Az,2)),     eye(nu),        zeros(nu,1);
          zeros(nr,size(Az,2)),     zeros(nr,nu),   eye(nr)             ];
Bz_ext = [Bz; eye(nu); zeros(1,nu)];
E = [Cz, zeros(size(Cz,1),1), -eye(nr)];

%% State constraints
x1_min = -1;    x1_max = 1.1;
x2_min = -0.2;  x2_max = 0.2;
% x_set = Polyhedron('lb',[x1_min;x2_min],'ub',[x1_max;x2_max]);

% % Polyhedron matrices of x:       Fx * x <= fx
Fx = [1, 0;-1, 0;0 1;0 -1]; 
fx = [x1_max;-x1_min;x2_max;-x2_min];

% % Extend to Polyhedron matrices of z:       Fz * z <= fz
Fz = Fx*[eye(nx), zeros(nx,nd)];
fz = fx;

% % Extend to Polyhedron matrices of z_ext:       Fz_ext * z_ext <= fz_ext
Fz_ext = Fz*[eye(nz), zeros(nz,nu), zeros(nz,nr)];
fz_ext = fx;

%% Control constraint
u_min = -0.1; u_max = 0.1;
% u_set = Polyhedron('lb',u_min,'ub',u_max);

% % Polyhedron matrices of u:       Gu * u <= gu
Gu = [1;-1]; 
gu = [u_max;-u_min];

% % u(k) = u(k-1) + du(k)
% % Gu * u(k) <= gu --> Gu * ( u(k-1) + du(k) ) <= gu
% % --> Gu*[zeros(nu,nz), eye(nu), zeros(nu,nr)] * z_ext + Gu * du(k) <= gu
% % --> Gz_ext* z_ext + Gu * du(k) <= gu

Gz_ext = Gu*[zeros(nu,nz), eye(nu), zeros(nu,nr)];

%% MPC data
Np = 12;
R = 1; Qe = 1;
Q = E'*Qe*E;

%% Stacked constraints
Uad = AdmissibleInputs(Az_ext,Bz_ext,Np,Fz_ext,fz_ext,Gu,gu,Gz_ext);

%% Quadratic Programming
Q_bar = blkdiag(kron(eye(Np-1),Q),Q);
R_bar = kron(eye(Np),R);
[A_bar,B_bar] = genConMat(Az_ext,Bz_ext,Np);
H = B_bar'*Q_bar*B_bar + R_bar;
options = mpcActiveSetOptions;
iA0 = false(size(Uad.b));

%% Simulation loop
Nsim = 350;
infea = 0;
% % Initialization
x0 = [0;0]; 
u = 0;
r0 = 1;

X_log = zeros(nx,Nsim);
X_log(:,1) = x0;

Xext_log = zeros(nz+nu+nr,Nsim);
x = x0;

% Observer gains calculation

% % -----    DO 1
% Qd = 1e-4; Rd = 1e-2; Nd = 0;
% [~,L,~] = kalman(sysd,Qd,Rd,Nd);
% L1 = L(1:ny); L2 = L(ny+1:end);
% x_hat = x; w_h = 0;

% % -----    DO 2
p = 0; w_h = 0; L = 0.5*[0, 5]; 
%
U_log = zeros(nu,Nsim);
U_log(:,1) = u;

R_log = zeros(nr,Nsim);
R_log(:,1) = r0;

W_log = zeros(1,Nsim);
What_log = zeros(1,Nsim);

Tcomp = tic;
for i = 1:Nsim
    % Disturbance & reference assignment
    if (i-1)*Ts < 1
        r = 0;
        w = 0;
    elseif (i-1)*Ts < 10
        r = r0;
        w = 0;
    elseif (i-1)*Ts < 25
        r = 0;
    else
        r = 0;
        w = 0.07;
    end
    % Observation
%     y = Cd*x + Dd*u;
%     [x_hat, w_h] = KalmanDisObs(x_hat,w_h,u,y,Ad,Bd,Cd,L1,L2);
    %
    [p,w_h] = ssObserver(L,p,w_h,x,u,Ad,Bd);
    %
    Xext_log(:,i) = [x;w_h;u;r];
    %
    q = B_bar'*Q_bar*A_bar*Xext_log(:,i);
    %
    [dU,exitflag] = mpcActiveSetSolver(H,q,Uad.A,Uad.b-Uad.B*Xext_log(:,i),Uad.Ae,Uad.be,iA0,options);
    %
    if exitflag <= 0
        du = 0; infea = infea + 1;
    else
        du = dU(1,nu);
    end
    u = u + du;
    U_log(:,i) = u;
    R_log(:,i) = r;
    W_log(:,i) = w;
    What_log(:,i) = w_h;
    X_log(:,i) = x;
    %
    x = Ad*x + Bd*(u+w);
end
Tcomp = toc(Tcomp);
Tcomp_ave = Tcomp/Nsim;
%% Plots
lw = 2;
TimeArray = 0:Ts:(Ts*Nsim);

figure(1);
fig1 = tiledlayout(4,1);

nexttile
plot(TimeArray(1,1:end-1),X_log(1,:),'LineStyle','-','Color','m','LineWidth',lw);
hold on
plot(TimeArray(1,1:end-1),R_log(1,:),'LineStyle','--','Color','k','LineWidth',lw);
ylabel('$x_1$','interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
plot(TimeArray(1,1:end-1),X_log(2,:),'LineStyle','-','Color','m','LineWidth',lw);
ylabel('$x_2$','interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
stairs(TimeArray(1,1:end-1),U_log(1,:),'LineStyle','-','Color','m','LineWidth',lw);
ylabel('$u$','interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
hold on
plot(TimeArray(1,1:end-1),What_log(1,:),'LineStyle','-','Color','m','LineWidth',lw);
plot(TimeArray(1,1:end-1),W_log(1,:),'LineStyle','--','Color','k','LineWidth',lw);
ylabel('$w$','interpreter','latex')
xlabel('Time (seconds)','interpreter','latex')
legend({'$\hat{w}$, estimate','$w$, true'},'interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

fig1.TileSpacing = 'compact';
fig1.Padding = 'compact';
set(gcf,'Units','points','position',[400, 50,800, 600])

%% Function helper
% 
function Uad = AdmissibleInputs(A,B,Np,Fz_ext,fz_ext,Gu,gu,Gx_ext)

[A_bar,B_bar] = genConMat(A,B,Np);

Uad.A = [blkdiag(kron(eye(Np-1),Fz_ext),Fz_ext)*B_bar;kron(eye(Np),Gu)+kron(eye(Np),Gx_ext)*B_bar];
Uad.b = [[kron(ones(Np-1,1),fz_ext);fz_ext];kron(ones(Np,1),gu)];
Uad.B = [blkdiag(kron(eye(Np-1),Fz_ext),Fz_ext)*A_bar;kron(eye(Np),Gx_ext)*A_bar];
Uad.Ae = zeros(0,size(B_bar,2));
Uad.be = zeros(0,1);
end

% 
function [A_bar,B_bar] = genConMat(A,B,Np)

A_bar = cell2mat(cellfun(@(x)A^x,num2cell((1:Np)'),'UniformOutput',false));
B_bar = tril(cell2mat(cellfun(@(x)A^x,num2cell(toeplitz(0:Np-1)),'UniformOutput',false)))*kron(eye(Np),B);

% % or using cell
%
% A_bar = cell(Np, 1);
% B_bar = cell(Np,Np);
% b0 = zeros(size(B));
% for i = 1:Np
%     A_bar{i} = A^i;
%     for j = 1:Np
%         if i >= j
%             B_bar{i,j} = A^(i-j)*B;
%         else
%             B_bar{i,j} = b0;
%         end
%     end
% end
% A_bar = cell2mat(A_bar);
% B_bar = cell2mat(B_bar);
%
end

% 
function [x_hat, d_hat] = KalmanDisObs(x_hat,d_hat,u,y,Ad,Bd,Cd,L1,L2)

x_hat = Ad*x_hat + Bd*u + Bd*d_hat + L1*(y-Cd*x_hat);
d_hat = d_hat + L2*(y-Cd*x_hat);
end

% 
function [p,d_hat] = ssObserver(L,p,d_hat,x,u,Ad,Bd)

p = p - L*((Ad-eye(size(Ad)))*x + Bd*u+Bd*d_hat);
d_hat = p + L*x;
end