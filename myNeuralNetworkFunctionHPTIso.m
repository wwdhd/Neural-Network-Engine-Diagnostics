function [Y,Xf,Af] = myNeuralNetworkFunctionHPTIso(X,~,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Auto-generated by MATLAB, 09-Dec-2024 17:48:47.
%
% [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
%
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = Qx13 matrix, input #1 at timestep ts.
%
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = Qx5 matrix, output #1 at timestep ts.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [-0.467728998446116;-0.765454557954978;-0.862982393895249;-0.726617923680114;-8.07258667552133;-1.93369960113561;-2.94246853860711;-0.687836296527042;-0.487715591472737;-0.750890242252019;-1.88593080775311;-0.0999983738116939;-5.51870946028399];
x1_step1.gain = [1.44249403654615;1.19508039502153;0.214974979312844;0.392505382574705;0.233379019822129;0.579644583105434;0.580284835738651;0.165338904254057;0.955519281572305;0.144727290574369;0.143069840987276;10.0041819242032;0.344352002176809];
x1_step1.ymin = -1;

% Layer 1
b1 = [-3.5285070019009414644;-1.3299660972666018655;1.3844378672406754127;1.2451463477659190371;-2.1348619513906341716;0.33798413735959076387;-2.1738510314625640873;-1.133791031271135008;-1.6601414980850961456;-2.0953931702711745544];
IW1_1 = [-0.16186888032045509966 -0.088232536122048996208 -0.39310186577265210373 -0.17337140247945628801 2.4604629967574709326 -0.016451951980176751644 -0.18651679254139030828 -0.7627037495387759547 -0.21521969289513201429 0.42099201244407818967 -0.24625259209434111063 0.091097248065890729096 0.63632800135796885321;0.17877117475940712543 -0.20925755119364339296 0.4140511407146283962 -0.24084111927244647444 -0.95251644788722855672 -0.025095307376125156051 -0.31096930063522354448 0.036393678558976121085 0.34433910244261944911 -0.38371191408184579519 -0.59734103798341398051 0.65970634760648283468 -0.51728927900084564762;0.18451374317186966523 0.25938082653022226376 0.98886326153466463929 0.56255143522468298301 -0.81955002368239793231 0.38157454406133695102 -0.17942922512024422899 -0.2314959320118820385 -0.54859524331453346502 -0.22788257052614924181 0.1406892862637831032 -0.75155633542823885307 -0.56597006358177281982;0.02527055701343461358 0.089928875631966720405 0.2202885233667216669 -0.057077341289914637623 -0.50980565147503553725 -0.73467733714753069574 0.67044514310808323021 0.42685987690732585254 -0.21361731471593600751 0.04406427234536837434 0.22493599655627496969 0.51615555135015833699 -0.87818953339527583424;-0.12376103416513981048 -0.041983058830953948193 -0.49121390207974868769 0.050871190681678488121 1.9797576162077135109 0.212408475866734614 0.087404126679851035231 0.051844588138580435799 -0.049692154304340201143 0.46164873141392598344 -0.30435905391010598109 -0.039133055632216529052 1.1553091248213449571;0.36038843428405092917 0.20103075041705104065 0.85765512073194238862 -0.19538130753453636368 -0.36814286799075901513 -0.82244916959660385736 -0.58479260015386591576 -0.69268514631949462679 0.15865482264953409408 -0.25799847388346147659 -0.29757085805575883564 0.46402831295819157997 0.0072086582234788020365;-0.11292209085895500709 0.044278959828072630978 0.15718958626785234367 0.29621487580220579039 1.4776223948373603534 0.1485118048626308529 0.52627900605620492325 0.37530575109826425084 0.088449353516960579902 -0.073946312697564284266 -0.08745859376524538209 -0.01750228733420367 1.0478188023148609087;-0.67845255992918707122 0.35556057290403009219 -0.3058935212021514638 0.066609426990924508472 -0.51988850422002763629 0.29125743605013326709 0.25167088421226629924 0.37625199921908381029 -0.59265768119075967757 0.11553236227984763074 -0.31503453666865183225 0.51426058520513207029 0.55461044055539643338;-0.173494484108530439 -0.14986269844758914882 -0.62301197500878324753 -0.35853522986978259013 0.36141346739813673405 -0.23241671571843397492 0.17245546942636624799 -0.54788195787232019107 0.42076075303049209619 0.42710611044293000926 0.095108944855307492316 0.70590740311572508503 0.22880443800360239504;-0.18129214561248443993 -0.06079478123694952163 0.028659814141060412085 -0.46973093980046443097 -0.10975034619207441056 -0.41238478701102410096 -0.46095399634575701109 -0.31636413635840626712 0.017227460539662085848 -0.57659125373960318228 0.082994756348953230285 -0.42910456141688491094 -0.66722989533219589919];

% Layer 2
b2 = -0.90283991972735866405;
LW2_1 = [-2.2963686107336980236 -0.033560890072849743071 0.41358247870963310566 0.12015764315277810415 1.3667009728132664126 -0.0070368169602980255514 -1.2664091475627934358 -0.046646695008141836336 0.77329840151533013604 0.11019759214549335047];

% Output 1
y1_step2.ymin = -1;
y1_step2.gain = 2;
y1_step2.xoffset = 0;
y1_step1.xrows = 5;
y1_step1.keep = 4;
y1_step1.remove = [1 2 3 5];
y1_step1.constants = [0;0;0;0];

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
    temp = mapminmax_reverse(a2,y1_step2);
    Y{1,ts} = removeconstantrows_reverse(temp,y1_step1);
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

% Remove Constants Output Reverse-Processing Function
function x = removeconstantrows_reverse(y,settings)
Q = size(y,2);
x = nan(settings.xrows,Q,'like',y);
x(settings.keep,:) = y;
x(settings.remove,:) = repmat(settings.constants,1,Q);
end