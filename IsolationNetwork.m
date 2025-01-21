function [Y,Xf,Af] = IsolationNetwork(X,~,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Auto-generated by MATLAB, 12-Dec-2024 17:23:30.
%
% [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
%
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = Qx13 matrix, input #1 at timestep ts.
%
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = Qx4 matrix, output #1 at timestep ts.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [-0.863042163354611;-0.898951495627522;-12.0786134381386;-4.26312013559145;-8.08256065417442;-1.72874849654628;-3.90424402237049;-3.28831133700203;-3.68648170280508;-1.79043181564066;-3.30186823623461;-0.099958696411262;-5.51446560271311];
x1_step1.gain = [1.12705916518384;1.11299940765634;0.0964720300253435;0.232251634754464;0.138299305273377;0.359586911545946;0.197160441780948;0.138027070555405;0.165014559023025;0.132601259884478;0.128059625199131;10.0029014327176;0.15440950308941];
x1_step1.ymin = -1;

% Layer 1
b1 = [4.9671163736982837378;-4.8579060065285579739;10.318590008828335769;3.4394411658913570662;5.8842712767323837042;-2.6875213072447929896;1.3553527028723788561;-2.0637697006459774229;0.63052001697690707172;-6.0130092938793566759;1.3961458434425439989;4.1061761706514499082;-3.7697976083689344051;-1.9848175418099489598;2.3282444099711216978];
IW1_1 = [0.1558262746609552496 0.22272647593415700884 -1.1719900503392863822 -1.0323552988321287316 0.00067306632648247988193 0.21005855716067073358 -3.8566327063805418973 -0.064039462579560002542 3.1601357584812368451 3.4130008004587089054 0.41420150845069592105 -0.23088273291283051281 1.4720948231814445784;-0.06548541964725525244 -0.017893632233571495072 0.73452415181091212926 -0.51330252004701648616 1.1337133030940775846 -0.69360175015812008947 3.0233581945321401641 -1.3496277351688015234 -1.3304478036250686479 -2.9940264722183984603 -0.14080909318343198811 0.06423043058310985709 -1.3280565403209134701;0.17537768031961922799 0.52007090733305472163 -21.50645001280109625 2.4900607510859806837 -8.0329277210991527625 -4.7635957899680825278 0.7108084371619105557 4.0689836062052133769 3.1541707325058032652 5.5910195721203370667 1.1266903429767087808 -0.40856922525974376237 -7.1028004047715462832;0.14025172207627692456 -0.087099797415750246365 5.3566282717622204501 2.4677516848464620125 -3.3607993065939574251 1.9945107546671536625 -2.116508860981824558 2.6854219707871891387 -1.2686780820170082684 1.6479348470825967432 -0.31322116552541123813 -0.015201417684765367727 1.7267807508684431816;0.13034965079733085158 -0.065802104987067122632 -1.4143918404829605162 0.73576650357893347287 -2.2775056227836336653 1.1464756428415159295 -0.96668889534498636529 2.6529631477643054183 -1.3994788670648736773 3.5411908221858561241 -0.5212085932509578079 0.027926619630718303722 5.6396873765360577835;-0.1382769069002663942 0.12533383974567252617 -4.9686145040107430049 -2.350518256935020478 1.9317529496587584337 -2.4359303817217941024 1.477717508351016118 -2.7009644276969129173 2.6555677738929510667 -1.2595166006832168026 0.39502129517502737599 -0.0078638602413797567547 -2.8494319863550416905;-0.0013281664824860740792 -0.11695758951476337151 1.1313973078345960221 0.21394098641799549898 6.4131829011282626141 5.6876142856163278694 -0.12840606908389684548 0.31722307214435474521 -3.0440340693572185415 0.075935029305387230503 -0.66542714902975474978 0.13384178918044475903 10.547265416710770225;0.045329877101252420701 -0.21314255289303221663 -7.2956662463005574892 1.6644256418916094642 0.63856813502668419158 1.7682144825717163172 3.5218571480114388983 0.057051224922290880159 -4.9658397324945786622 -1.9810752748124897948 -0.87327779687124729158 0.10923835750435428871 -2.7970596586045393117;0.88361737435291287657 0.057981043521953413988 10.389907303847424558 4.5743845347409184399 -12.901711924970241441 3.5288039778558202286 -1.955606928194699945 11.196504960710306875 -23.551688192512177267 1.4937957629674398774 -4.9489674270379904186 0.93123793226530438005 13.848017431189374449;0.42169160009171013925 -0.32363442383919238665 36.719050108333505023 -1.5770173831525320374 -1.0827034323738362609 -1.2751021662381105681 -0.9531030767705983564 -0.43639670375164496541 -1.4344675044309556089 -0.064331989687669649536 -0.41758272419787523244 0.31291870853048098233 8.5397812121812606279;0.013728741232241515116 -0.099415521500408901567 1.4586153615932504479 0.32243881706473348503 5.2036526239566027741 4.9481730213142673946 -0.04827354367488713438 0.33095960429695958638 -2.5790872700173141929 -0.05117661931202876352 -0.58343379800968730997 0.11693135914120915408 9.665939384650595656;-0.08844700999048869805 0.022327641047960165854 -0.37355176030680492527 -1.1143113965316595593 -0.444529006529108206 0.90760761081830620967 -1.2405003867632777848 0.37780196890111938934 1.3151300091357613908 1.5020541869304304683 0.83249400115372051978 -0.064187999348693597357 0.92639208606085521325;-0.0014528870440881115622 -0.028869832555602950586 -2.7318518466473880046 -0.80297387210647186429 0.92532008977788593462 -2.3175812219647240475 1.0974230599615337756 -0.71790003700249138685 -1.8139131365002378882 -0.7196508641254562022 0.016968501091201390607 -0.064890043717862333494 -4.1562842623056894809;0.030298013461916802058 -0.21340275741264524978 -7.1712239115807951961 1.6894680187458577247 0.63513927333122577323 1.6928865762219382773 3.4055700135229898251 0.16769635404131855161 -4.9994872423572518372 -1.9752703140839531493 -0.82264802234222633714 0.098937937506114362329 -2.8733295387751995875;-0.10224221512922104793 -0.0093355189517701046842 -0.15991841798086325888 -0.218873027792697461 0.3368326437213858271 0.29928927715146946831 -0.46937612678227291774 0.28789204019101199972 0.84195361282719627383 0.031878918519218230054 0.66770781367951792529 0.12060577602373723227 -0.27555738125139533912];

% Layer 2
b2 = [-0.6324579726673620339;-1.5218795288646269626;0.27358566552067714595;-0.11943931544962423097];
LW2_1 = [-0.80842981598480812 -0.024163283335745425195 -0.040893545051841964644 -0.041745678433613445102 -0.21050219567939718046 -0.049581758916459606201 0.060968190056785900865 0.73405810273460325721 1.0635301036403828068 -1.1122090281820979918 -0.1003593174097485996 0.66833383044569527609 -0.032896759740236633518 -0.80759714209515220951 -0.10703680734057241519;1.9694077868975068402 1.6782222507261430611 -0.10591979096420933715 1.518335911591566223 1.6003993567559919864 1.6362007230599537877 -3.1818031585387620552 -1.303699137869002378 0.0074151632575235638944 1.0111109473011044013 4.0807908088795334933 -0.89898419102116766677 0.78129439397457267624 1.4332865302455799483 0.55010610962505679389;-1.1620095253940663227 -1.5139015613894963863 0.15866561372784018791 -1.4668662279647071145 -1.4665734620065555838 -1.5504980437662276849 3.1010930932895255374 0.68514915587183866652 -0.027052337109679917143 0.095349846324875225934 -3.9483958518737662935 0.28946396417429953729 -0.76069769260694086732 -0.76033298590352560442 -0.44648652207249550639;0.00020976649398295882376 -0.14059087845939499672 -0.01191311687418095816 -0.0096335592358078071806 0.076490405024987911942 -0.036033879078836691945 0.019707083999025421656 -0.1148019299747989469 -1.0438564143519062366 0.005678536956540757713 -0.032050454174112036343 -0.057890347599134757417 0.01229585501206545764 0.13389784679082106233 0.0032910756163796878182];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = [2;2;2;2];
y1_step1.xoffset = [0;0;0;0];

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