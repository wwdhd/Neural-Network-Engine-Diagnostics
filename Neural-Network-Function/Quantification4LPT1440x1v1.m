function [Y,Xf,Af] = Quantification4LPT1440x1v1(X,~,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Auto-generated by MATLAB, 22-Jan-2025 12:13:16.
%
% [Y] = Quantification4LPT1440x1v1(X,~,~) takes these arguments:
%
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = Qx13 matrix, input #1 at timestep ts.
%
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = Qx2 matrix, output #1 at timestep ts.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [-0.84905506094289;-0.89661207983156;-11.8900564136425;-4.27206819854083;0.117334445616086;-0.477452394268215;-3.93907360901864;-3.41204081619662;0.44191074414151;-1.87637380752854;-0.691499300063549;-0.0999911658899677;0.481243099714201];
x1_step1.gain = [1.55105753151625;1.20876767498217;0.177016830453653;0.426837026831518;0.318106768607705;1.01183097308742;0.202423161580597;0.241345617441474;0.255153784970049;0.211806501738349;0.174628547902164;10.0099470950088;0.286382613913033];
x1_step1.ymin = -1;

% Layer 1
b1 = [-1.417767800131442435;-1.1636401893981587197;-1.1142275417169094087;0.10900006543030776873;0.18998161084903039675;0.0157326253890928483;0.52110334615627040833;1.001553926691583829;0.2556532205006366576;1.4051091532076862567];
IW1_1 = [-0.001111349244602138648 -0.30320780159328847292 0.72773622441834173724 -0.32416923453464296934 -0.19744832119376398705 0.18118197953608805584 -0.30282966000358424452 -0.50453559663778402289 -0.079560272532807715939 0.25547793454280975922 -0.31907504730285124461 -0.15272424380642896091 -0.63277253648676434761;0.7038454941563248024 -0.3997610138286019632 0.075928652177309735594 0.27165038361018389867 0.29296357540137646369 -0.33942581111674113847 -0.66239026972874970323 -0.019775444972161635576 0.62519741564408315426 0.18906501875304312921 0.73765964874603251467 -0.29239033919195300815 -0.19146967670371631276;0.25783583691025818707 -0.0084290233513566042989 0.26607954373002828952 0.22740467599006664035 -0.17147597960272864537 -0.41958731812126948135 0.42701336772898007554 -0.42580964100302043374 -0.51849928962628899498 -0.67788788229196683677 0.42687429963106260367 0.55026006903768553968 0.13638256007979152051;-0.0077544628194885700453 0.01182643884860720869 0.17305509018995468562 -0.0094191183191021498472 0.0048265916110138721781 0.0085133282157751240249 0.065855296077372407759 0.047347722940921786738 -0.090782598987369289567 -0.086324406303116746342 0.012994222893957661086 0.00031445687014532109835 0.7853097839244782552;-0.013567754675798664871 -0.098032005734285582177 0.23335553216303961399 -0.036038809764284245041 -0.015506470335289956886 0.11784181497303192199 -0.60119257173748696932 0.042590065420192509127 0.21851156132193375048 -0.028614791901716521427 -0.40096110906222709458 -0.0086337271783873438535 -0.33757709587491258052;0.064844862413444892546 0.051229302540365360963 0.16017533137463943471 0.04452907196185259403 -0.06118548364761452335 -0.058042596606234050927 0.014102024895082167569 -0.086782101574365999586 -0.21281284728476637658 -0.20067891329147929946 0.20616158749821916762 -0.029176426363415251114 0.33071298895999157175;0.56442231178447188622 -0.16440356094352528982 -0.20764819017887689578 0.20672648368289409104 0.22031373632850964617 -0.22563050831829944975 -0.29545486021299965929 -0.40491554180257893769 0.61222338879421323021 0.526876508877364369 0.65665421785023736501 -0.18157123654426568504 -0.59183789157762878297;0.19460596956569539695 -0.74788551760229504684 0.58842474532161836986 0.026559519243069304451 0.49513797497240513712 -0.21971541219958706792 0.03207414859386760464 0.9000277174617770104 -0.72472641241776114551 0.046144147415280910296 -0.46734721188083183652 0.10135416158175267198 -0.34826357381826450466;-0.050130913580003995389 0.025363276233848673635 0.406042045882648861 -0.023404619080273969534 0.012842405361281504159 0.060686006750040703828 0.25101821733197698938 0.16267661864721510989 -0.059171331766559025211 -0.024680100931548244747 -0.051372743316830331295 0.0094474049112623498248 0.31128363987646384681;0.089430902164480047256 0.040957040351377212162 -0.48466284662123931826 0.1392977040718520465 -0.41678680096901865593 0.088007573246325507266 -0.68604401293027383879 -0.79681571979503240666 0.90260575038316759322 -0.030070566305087507303 -0.18359302744902539217 -0.051869119699602887763 0.68796558195468027197];

% Layer 2
b2 = [-0.028668385084369314708;0.24036329411549639534];
LW2_1 = [-0.048773317329353998995 -0.0071384458750963367471 -0.018761656355044332295 1.7855271679687640951 0.00047685200313317584186 -0.32416683876962015809 -0.0048036506964627074048 -0.021461673271600263668 -0.77773879688733082993 0.085375973657323273036;0.16444277903811285735 -0.16864992277556276057 0.038906976949751509443 -0.29475473922882822198 0.74113791020636332618 1.4848562988011380082 -0.25560073346568662478 0.052634940362715335049 -1.0701626355739313201 -0.010937958581485350551];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = [0.363636363636364;0.363636363636364];
y1_step1.xoffset = [0.5;-6];

% ===== SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if ~isCellX
    X = {X};
end

% Dimensions
TS = size(X,2); % timesteps
if ~isempty(X)
    Q = size(X{1},1); % samples/series
else
    Q = 0;
end

% Allocate Outputs
Y = cell(1,TS);

% Time loop
for ts=1:TS

    % Input 1
    X{1,ts} = X{1,ts}';
    Xp1 = mapminmax_apply(X{1,ts},x1_step1);

    % Layer 1
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*Xp1);

    % Layer 2
    a2 = repmat(b2,1,Q) + LW2_1*a1;

    % Output 1
    Y{1,ts} = mapminmax_reverse(a2,y1_step1);
    Y{1,ts} = Y{1,ts}';
end

% Final Delay States
Xf = cell(1,0);
Af = cell(2,0);

% Format Output Arguments
if ~isCellX
    Y = cell2mat(Y);
end
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin);
x = bsxfun(@rdivide,x,settings.gain);
x = bsxfun(@plus,x,settings.xoffset);
end