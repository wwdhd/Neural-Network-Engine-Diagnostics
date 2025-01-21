function [Y,Xf,Af] = myNeuralNetworkFunctionHPTQua(X,~,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Auto-generated by MATLAB, 09-Dec-2024 18:07:23.
%
% [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
%
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = Qx13 matrix, input #1 at timestep ts.
%
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = Qx10 matrix, output #1 at timestep ts.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [-0.467728998446116;-0.765454557954978;-0.862982393895249;-0.726617923680114;-8.07258667552133;-1.93369960113561;-2.94246853860711;-0.687836296527042;-0.487715591472737;-0.750890242252019;-1.88593080775311;-0.0999983738116939;-5.51870946028399];
x1_step1.gain = [1.44249403654615;1.19508039502153;0.214974979312844;0.392505382574705;0.233379019822129;0.579644583105434;0.580284835738651;0.165338904254057;0.955519281572305;0.144727290574369;0.143069840987276;10.0041819242032;0.344352002176809];
x1_step1.ymin = -1;

% Layer 1
b1 = [-1.7524304674980426544;0.75360835078290455691;-0.094549892949409672371;-0.22097095269130492823;-0.49890540939244087948;-0.21241064187047489775;-0.12333121765336792219;-0.13700948282633240716;3.1049701109384972852;-1.1055021285836192924];
IW1_1 = [0.035829783896510418451 0.12541567262406286276 0.55073242896148011383 -0.050233852327277207783 -0.7067452729025024416 -1.0265554847494238189 0.03862072551453382685 -2.7565120930895754547 0.063397024965027901078 1.1654371030977657142 0.051483423998055943827 0.0085664854323885541998 -0.33153599773326031253;0.021028650153994077332 0.28632196246659358252 0.99064143398906490123 0.62933330624058780867 -1.0187635452097578703 -0.037290764726500405524 1.4602246278018864434 -0.26721663149544089277 0.034270952365901101444 1.6352837959862422945 -0.15476008621194745074 0.46984458358304781456 2.5763332979597954875;-0.10853743406848073705 0.0170081816548278697 -0.31277642914534342644 0.31285115223865783696 0.21211168680713299306 0.39697921033229388588 0.8008101684721062874 1.9103837191242321047 -0.1614727152881337835 -0.10525146401510364347 0.15375596126365612126 0.07360427303459214643 0.45753083477377870514;0.0059470430280234316128 -0.0072297555336454741998 -0.12629072083261660975 0.0011945719199541503588 -0.0011311123588207219227 0.042638653159369398205 -0.026943406849077991549 0.092322899524343290945 0.0014372415223798463198 0.21088649570480422213 -0.01847077652737931508 0.0049002746051747076914 -0.00033201472142188430403;-0.018250516607049231677 -0.19785763138020678875 -0.37941804145777424662 0.18571441315463199651 -1.0280944690397788577 0.4538100249885666515 0.66618507439343088716 -1.042493390646291429 0.29096149903844686913 -0.74289819462475559231 -0.61119764907159002121 0.36675980852719836101 -2.1642179942816652805;0.051055259837563572722 0.031254530083515363881 0.85149379315984419225 -0.10598262384936439628 0.60350841020559897743 -0.23486402199388534129 0.022394734065559460112 -1.2640038391887462588 0.13747773469941423907 -0.50379374707801438671 -0.047807442949256380438 0.065853270351579090702 -1.6422149078081291762;-0.02735217428586149413 0.039501630053533291764 0.6674083353980541089 0.0011683759459737222408 -0.083607537350524133157 -0.21284992947503436622 0.11104750199762007667 -0.36522292004974077351 -0.0018426268766354547517 -0.87714169325933066812 0.11325086800815814536 -0.028746588494653189938 -0.57922544722677493656;-1.8990873936803482902 0.43809124312612091057 -5.8567584623923210074 3.148806190636289859 -1.4083998151809342669 -0.31762124175130485026 0.37786013095439302534 1.7807655856234843039 0.25279316980176175234 2.2104880929609231899 -0.27529454097065031437 -1.5280239073303421016 1.3900785803873896285;0.093371239077442116927 1.1832715871131982421 2.7236835435267305883 1.7772479808408279123 -2.5253363354514677397 -2.2044798591406480348 1.1331862842748472442 -3.459663319330587683 -1.0474029157297413128 -3.5356374367871286069 -2.2207823530036212567 1.4211344833961632173 -1.4638614269018423997;-0.19213170381630456651 -0.53972267650496097424 1.5407261915159200516 1.4687615990459887971 0.65471639982717411144 -0.57058690395913391757 -1.2532983811721187806 -4.4528039583838170401 -0.020699625967238059754 1.0885978754316745221 -2.0607792586041808747 0.24587760001001357502 -1.4234780898423116113];

% Layer 2
b2 = [1.5688827960084552604;-1.0938902776230794434];
LW2_1 = [0.23243285889790757381 0.12184066734796582254 -0.4815833739134684488 6.3763222096516880555 0.25044128906220036734 -0.99695981107676756583 2.017952704358450422 -0.031811290192387264852 0.015813915318146257172 -0.035939366494626014126;0.13155936763441961057 0.069715255327192512946 -0.26797869471567886679 -4.6287074698710171106 0.10992729917155784347 -0.52084326913602230213 -0.54020562542310135434 -0.010578794381140890421 -0.00010652971432703118356 -0.024200732801506634473];

% Output 1
y1_step2.ymin = -1;
y1_step2.gain = [0.333333333333333;0.307692307692308];
y1_step2.xoffset = [0;-6];
y1_step1.xrows = 10;
y1_step1.keep = [7 8];
y1_step1.remove = [1 2 3 4 5 6 9 10];
y1_step1.constants = [0;0;0;0;0;0;0;0];

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

% Sigmoid Symmetic Transfer Function
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
