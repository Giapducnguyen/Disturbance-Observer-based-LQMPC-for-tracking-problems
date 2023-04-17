% % ACC - 2nd-order dynamics - with disturbance observer
% % Coded using the Reference Tracking formulation 2 of
% % Module 5 - Introduction to MPC - Kolmanovsky

% % Note 1: Qe and L are the most important tuning parameters
% % Note 2: It might be impossible to apply Terminal weight and constraints
% % Note 3: Since the problem is modelled as a TRACKING problem, it might
% %         be impossible to apply Tube-MPC
% % Note 4: It showcases the use of a Disturbance observer in MPC

clear all
close all
clc

%% Discrete-time model
Ts = 0.1; % sampling period

Ad = [1, Ts;0, 1];
Bd = [0.5*Ts^2; Ts]; Bu = -Bd;
Cd = [1, 0;0, 1];
Dd = 0;

nx = size(Ad,2);    % number of states
nu = size(Bd,2);    % number of inputs
ny = size(Cd,1);    % number of outputs
nr = 2;             % number of references
nd = 1;             % number of disturbances

%% Extend state to include disturbance estimate \hat{w} 
% % state z = [x(k); \hat{w}(k)];

Az = [Ad,                       Bd;
      zeros(nd,size(Ad,2)),     eye(nd)];
Bz = [Bu; zeros(nd,size(Bu,2))];
Cz = [Cd, zeros(size(Cd,1),nd)];

nz = size(Az,2);

%% Extend state to include control increments du and reference r
% % state z = [z(k); u(k-1); r(k)];

Az_ext = [Az,                       Bz,             zeros(size(Az,1),nr);
          zeros(nu,size(Az,2)),     eye(nu),        zeros(nu,nr);
          zeros(nr,size(Az,2)),     zeros(nr,nu),   eye(nr)             ];
Bz_ext = [Bz; eye(nu); zeros(nr,nu)];
E = [Cz, zeros(size(Cz,1),1), -eye(nr)];

%% Time data
Tsim = 40; % simulation time
Nsim = floor(Tsim/Ts); % simulation steps

%% State constraints
%
% % velocity constraints
min_v = 0; % not negative in highways
max_v = 36.11; % [m/s] == 130 km/h; imposed by law
min_delta_v = -max_v;

% % safe distance constraint
max_sensing_range = 115;
tau_h = 1.4; % constant time headway % 0.5
d0 = 10; % stopping distance
max_delta_d = max_sensing_range;%-d0;
%
x1_min = d0;    x1_max = max_delta_d;
x2_min = min_delta_v;  x2_max = max_v;
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
%
% % Input constraints
gacc = 9.81;
u_min = -0.5*gacc;
u_max = 0.25*gacc;
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
Np = 10;
R = 1; Qe = diag([1, 1]); %diag([0.1, 5]);
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

infea = 0;
% % Initialization
% ACC vehicle
x_fInit = 10;
v_fInit = 30;
% a_fInit = 0;
X_f = zeros(nx,Nsim);
X_f(:,1) = [x_fInit;v_fInit];

% Preceding vehicle
x_pInit = 100;
v_pInit = 10;
% v_pEnd = 20;
a_p = u_max;
X_p = zeros(nx,Nsim);
X_p(:,1) = [x_pInit;v_pInit];

% Inter-vehicle
x0 = X_p(:,1)-X_f(:,1);
u_f = 0;
r0 = [d0 + tau_h*X_f(2,1);0];

X_log = zeros(nx,Nsim);
X_log(:,1) = x0;

Xext_log = zeros(nz+nu+nr,Nsim);
x = x0;

% Observer gains calculation
p = 0; w_h = 0; L = 0.1*[2, 10]; %0.1*[2, 10]; 
%
U_log = zeros(nu,Nsim);
U_log(:,1) = u_f;

R_log = zeros(nr,Nsim);
R_log(:,1) = r0;

W_log = zeros(1,Nsim);
What_log = zeros(1,Nsim);

Tcomp = tic;
for i = 1:Nsim
    % Disturbance & reference assignment
    if i >= floor(10/Ts) && i < floor(20/Ts)
        a_p = 0;
    elseif i >= floor(20/Ts) && i < floor(30/Ts) && X_p(2,i) >= 1.41625
        a_p = u_min;
    elseif i >= floor(30/Ts) || X_p(2,i) <= 1.41625
        a_p = 0;
    end
    r = [d0 + tau_h*X_f(2,i);0];
    % Observation
    [p,w_h] = ssObserver(L,p,w_h,x,u_f,Ad,Bd,Bu);
    %
    Xext_log(:,i) = [x;w_h;u_f;r];
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
    u_f = u_f + du;
    U_log(:,i) = u_f;
    R_log(:,i) = r;
    W_log(:,i) = a_p;
    What_log(:,i) = w_h;
    X_log(:,i) = x;
    %
%     x = Ad*x + Bu*u_f + Bd*w; % % State update
    %
    X_p(:,i+1) = SingleVehicleDynamics(X_p(:,i),a_p,Ts);
    X_p(2,i+1) = max(0,X_p(2,i+1));
    % ACC vehicle
    X_f(:,i+1) = SingleVehicleDynamics(X_f(:,i),u_f,Ts);
    %
    x = X_p(:,i+1)-X_f(:,i+1);
end
Tcomp = toc(Tcomp);
Tcomp_ave = Tcomp/Nsim;
%% Plots
lw = 1.5;
TimeArray = 0:Ts:Tsim;

figure(1);
fig1 = tiledlayout(4,1);

nexttile
plot(TimeArray(1,1:end-1),X_log(1,:),'LineStyle','-','Color','m','LineWidth',lw);
hold on
plot(TimeArray(1,1:end-1),R_log(1,:),'LineStyle','--','Color','k','LineWidth',lw);
ylabel('$\Delta d (m)$','interpreter','latex')
legend({'Actual', 'Desired'},'interpreter','latex','Location','best')
title('Inter-vehicle Distance','interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
plot(TimeArray(1,1:end-1),X_log(2,:),'LineStyle','-','Color','m','LineWidth',lw);
ylabel('$\Delta v (m/s)$','interpreter','latex')
title('Velocity Difference','interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
stairs(TimeArray(1,1:end-1),U_log(1,:),'LineStyle','-','Color','m','LineWidth',lw);
hold on
stairs(TimeArray(1,1:end-1),u_min*ones(size(TimeArray(1,1:end-1))),'LineStyle','--','Color','k','LineWidth',lw);
stairs(TimeArray(1,1:end-1),u_max*ones(size(TimeArray(1,1:end-1))),'LineStyle','--','Color','k','LineWidth',lw);
ylabel('$u_f (m/s^2)$','interpreter','latex')
legend({'Command','Constraints'},'interpreter','latex','Location','best')
title('ACC Acceleration as Input','interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
hold on
plot(TimeArray(1,1:end-1),What_log(1,:),'LineStyle','-','Color','m','LineWidth',lw);
plot(TimeArray(1,1:end-1),W_log(1,:),'LineStyle','--','Color','k','LineWidth',lw);
ylabel('$a_p (m/s^2)$','interpreter','latex')
xlabel('Time (seconds)','interpreter','latex')
legend({'Estimated','True'},'interpreter','latex','Location','best')
title('Preceding Vehicle Acceleration as Disturbance','interpreter','latex')
box on
grid on
ax = gca; ax.FontSize = 14;

fig1.TileSpacing = 'compact';
fig1.Padding = 'compact';
set(gcf,'Units','points','position',[400, 50, 800, 600])

figure(2);
fig2 = tiledlayout(2,1);

nexttile
plot(TimeArray,X_f(1,:),'LineStyle','-','Color','m','LineWidth',lw);
hold on
plot(TimeArray,X_p(1,:),'LineStyle','--','Color','k','LineWidth',lw);
ylabel('$x (m)$','interpreter','latex')
title('Absolute Position','interpreter','latex')
legend({'ACC','Preceding'},'interpreter','latex','Location','best');
box on
grid on
ax = gca; ax.FontSize = 14;

nexttile
plot(TimeArray,X_f(2,:),'LineStyle','-','Color','m','LineWidth',lw);
hold on
plot(TimeArray,X_p(2,:),'LineStyle','--','Color','k','LineWidth',lw);
xlabel('Time (seconds)','interpreter','latex')
ylabel('$v(m/s)$','interpreter','latex')
title('Absolute Velocity','interpreter','latex')
legend({'ACC','Preceding'},'interpreter','latex','Location','best');
box on
grid on
ax = gca; ax.FontSize = 14;

fig2.TileSpacing = 'compact';
fig2.Padding = 'compact';

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
function [p,d_hat] = ssObserver(L,p,d_hat,x,u,Ad,Bd,Bu)

p = p - L*((Ad-eye(size(Ad)))*x + Bu*u+Bd*d_hat);
d_hat = p + L*x;
end

% % Individual vehicle dynamics
function x_n = SingleVehicleDynamics(x_c,u_c,ts)
A = [1  ts;
     0   1]; % state matrix
B = [0.5*ts^2; ts]; % control matrix

x_n = A*x_c + B*u_c;
end