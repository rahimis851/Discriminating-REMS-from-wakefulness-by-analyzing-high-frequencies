
        %% Information
% This algorithm is derived from the article, titled "Discriminating rapid
% eye movement sleep from wakefulness by ...
% analyzing high frequencies from single-channel EEG recordings in mice",
% conducted by Rahimi et al. (2023), a collaboration between Medical
% Unviersity of Innsbruck, Asutria, and Technical University of Munich,
% Germany. For further information and questions, you can contact the first author,
% Dr.med.Sadegh Rahimi (sadeghrahimiemail@gmail.com)

        %% Preparation
clc
clear all
close all

        %% Inputs

% As you can find in the manuscripte, we had recording from three locations, including electrode on the motor, sensory, and visual cortex. 
% In our hands, the best result derived from visual cortex
DataSource=3;                            % 1 means Motor Cortex, 2 means Sensory Cortex, 3 means Visual Cortex
MOUSE=1;                                 % in the study, we had 23 hours recording from from 9 mice. For the demo algorithm, we put only one hour EEG from the first mouse.   
SR=1000;                                  % Sampling Rate in Hertz
TimeSection=4;                            % In our experiments, we assessed the vigilance states in 4 seconds bins
Iteration=1;                             % Number of Iteration in Feature Extraction Section
    % Feature Extraction
% We selected nine sub-frequencies ([aFreq bFreq])
aFreq=[0.1;4;8;13;30;80;120;200;350];
bFreq=[4;8;13;30;80;120;200;350;500];

    % Outliers                           
% Artifacts, in particular, spike-like artifacts, can have broadband effects. To avoid them, we removed the outliers

ReplacingMethod='linear';    % 'center' , 'clip' , 'previous' , 'next' , 'nearest' , 'linear' , 'spline' , 'pchip' , 'makima'
DetectingMethod='grubbs';  % 'median' (default) , 'mean' , 'quartiles' , 'grubbs' , 'gesd'
    % Orders
ArtifactRemoval=1;                                        % Removing Artifacts according to std
Plot1=0;                                                          % Effects of Artifact Removal Algorithm
Plot2=0;                                                          % Single-sided Amplitude Spectrum After FFT
AveragePowerUsingPSD=0;                            % Average Power Computed by Integrating the Power Spectral Density (PSD) Estimate
% We imported this model (PSD) just for REM vs Wakefulness
MinMaxNormalizationMethod=0;
% Detecting the results of two stages (1 means on, 0 means off) - Default --> REM versus Wakefulness
NonREMvsREM=1;                                           % NonREM versus REM
NonREMvsWakefulness=0;                              % NonREM versus Wakefulness
BayesianClassifierMethod=1;                           % Kernel Naive Bayesian Method for Classification
AllFrequecyRanges=0;                                      % 1 means Feature Extracting using all Frequency Ranges
SpecifiedFreqRange=1;                                     % Between 1 to 9 (See aFreq or bFreq) - If AllFrequecyRanges=0

    % Notification, Which dataset has been selected
if (DataSource==1)                                      % Motor Cortex
    disp(['Motor Cortex of Mouse number ',num2str(MOUSE),' will be tested']);
elseif (DataSource==2)                                % Sensory Cortex
    disp(['Sensory Cortex of Mouse number ',num2str(MOUSE),' will be tested']);
elseif (DataSource==3)                                % Visual Cortex
    disp(['Visual Cortex of Mouse number ',num2str(MOUSE),' will be tested']);
else
    error ('ERROR, Please Select the Data Source')
end

        %% EEG File Name (of Mouse 1)
if (DataSource==1)
    EEGName='M2LJ01_M1';
elseif (DataSource==2)
    EEGName='M2LJ01_S2';
else
    EEGName='M2LJ01_V3';
end

    % Loading EEG
RecordedDataEDF=[EEGName,'.mat'];
[FName,~,~] = fileparts(which(mfilename));
FileName=[FName,'\',RecordedDataEDF];
load(FileName)

    % EEG naming
if (DataSource==1)
    EEG=M2LJ01_Baseline_Ch1.values;             % Motor Cortex
    clear RecordedDataEDF FileName M2LJ01_Baseline_Ch1
elseif (DataSource==2)
    EEG=M2LJ01_Baseline_Ch2.values;             % Sensory Cortex
    clear RecordedDataEDF FileName M2LJ01_Baseline_Ch2
else
    EEG=M2LJ01_Baseline_Ch3.values;               % Visual Cortex
    clear RecordedDataEDF FileName M2LJ01_Baseline_Ch3
end

    % Loading labels  (of Mouse 1)           
% We had excels files as groundtruth labels, 1 for Wakefulness, 3 for NREMS
% and 7 for REMS
WakefulnessLabel=1;
NonREMLabel=3;
REMLabel=7;
ExcelName='M2LJ01 scoring vector.xlsx';                      
Sheet=1;
Column='D2:D20701';
Labels=xlsread(ExcelName,Sheet,Column);

        %% Artifact Removal                   
% Our recordings had movement artifacts, which were eliminated by defining a threshold and excluding values above and below it. 
% In our case, we prefered not only exclude the values above and below the threshold, but also 50
% miliseconds around the peak, which is defined as "TimeDuration" below.

Threshold=1.5;                                                   % Threshold Level for Artifact Removing in mili volt
Factor=5;                                                            % Threshold Factor for Artifact Removing in mili volt
WindowRemoving=100;                                    % Length of step for Artifact Removing in mili second
TimeDuration=50;                                              % Time duration for Artifact Removing in mili second - Before and After
OriginalEEG=EEG;                                               % Orginal EEG Data
if (ArtifactRemoval==1)
    StartRemoving=100;
    EndRemoving=StartRemoving+WindowRemoving;
    for ii=1:round(length(EEG)/WindowRemoving)-WindowRemoving
        STD=std(EEG(StartRemoving:EndRemoving));       % Standard deviation of the Cutted signal
        n=1;
        for jj=1:WindowRemoving
            if abs(EEG(StartRemoving-1+jj))>(Factor*STD) || abs(EEG(StartRemoving-1+jj))>Threshold    % Criteria
                Before=jj-TimeDuration+StartRemoving;
                After=jj+TimeDuration-1+StartRemoving;
                EEG(Before:After)=0;
                KK(n)=jj;n=n+1;
            end
        end
        
        StartRemoving=StartRemoving+WindowRemoving;
        EndRemoving=EndRemoving+WindowRemoving;
    end
end

        %% Plot --- Effects of Artifact Removing
if (Plot1==1)
    figure;
    plot(OriginalEEG,'b');hold on
    plot(EEG,'r');axis tight
    xlabel('Time (Sample)');ylabel('Value (Micro volt)');
    legend('Before','After')
    title('Effects of Artifact Removing')
    
    figure
    s=length(EEG)/SR;Min=s/60;TimeMin=[];TimeMin=linspace(0,Min,length(EEG));
    plot(TimeMin,OriginalEEG,'b');hold on
    plot(TimeMin,EEG,'r');axis tight
    xlabel('Time (Min)');ylabel('Value (Micro volt)');
    legend('Before','After')
    title('Effects of Artifact Removing')
end

        %% Noise Elimination using Notch Filter (Mains Frequency and its harmonics) 
% in Europe, it will be 50 Hz, in USA, 60 Hz
FilteredData=EEG;
if (mod(bFreq(end),50)==0)
    LastHarmonic=floor(bFreq(end)/50)-1;
else
    LastHarmonic=floor(bFreq(end)/50)+1;
end
for ii=1:LastHarmonic
    wo = (50*ii)/(SR/2);
    bw=wo/35;
    [b,a] = iirnotch(wo,bw);
    FilteredData=filtfilt(b,a,FilteredData);
end

        %% Plot - Single-sided Amplitude Spectrum After FFT
if (Plot2==1)
    L=[];L=length(FilteredData);
    T=[];T=1/SR;
    NFFT=[];NFFT = 2^nextpow2(L);                    % Next power of 2 from length of Y
    X=[];X = fft(EEG,NFFT)/L;                                 % Raw Data
    Y=[];Y = fft(FilteredData,NFFT)/L;                    % Filtered Data
    f=[];f = SR/2*linspace(0,1,NFFT/2+1);
    
        % Plot Single-sided Amplitude Spectrum
    figure;
    plot(f,2*abs(Y(1:NFFT/2+1)));axis tight
    title('Single-Sided Amplitude Spectrum of Data After FFT');
    xlabel('Frequency (Hz)');ylabel('Amplitude');
    legend('After Filtering');
    figure
    plot(f,2*abs(X(1:NFFT/2+1)));axis tight
    title('Single-Sided Amplitude Spectrum of Data Before FFT');
    xlabel('Frequency (Hz)');ylabel('Amplitude');
    legend('Before Filtering');
    clear L T NFFT Y f;
end

        %% Feature Extraction
    % Sections of EEG
Window=TimeSection*SR;                                                  %   In our experiments, we assessed the vigilance states in 4 seconds bins
Start=1;
End=Window;
NumberOfSection=length(FilteredData)/Window;
clear EEG

for ii=1:NumberOfSection
    Section=[];Sections=FilteredData(Start:End);
    Start=End+1;
    End=End+Window;
    
            %% Feature Extraction Algorithm
    if (AllFrequecyRanges==1)                           % If you want to use all sub-frequencies
        Fea=[];
        for jj=1:length(aFreq)
            if (AveragePowerUsingPSD==1)
                    % Periodogram Power Spectral Density
                Pxx=[];[Pxx,~]=periodogram(Sections,[],[aFreq(jj) bFreq(jj)],SR,'power');
                Fea=[Fea,Pxx];
            else
                    % Average Power in the Specified Frequency Range
                Features(ii,jj)=bandpower(Sections,SR,[aFreq(jj) bFreq(jj)]);
            end
        end
        if (AveragePowerUsingPSD==1)
            Features(ii,:)=Fea;
        end
    elseif (AllFrequecyRanges==0)                    % If you want to use just a specified sub-frequecy
        if (AveragePowerUsingPSD==1)
                % Periodogram Power Spectral Density
            Pxx=[];[Pxx,~]=periodogram(Sections,[],[aFreq(SpecifiedFreqRange) bFreq(SpecifiedFreqRange)],SR);
            Features(ii,1) = Pxx;
        else
                % Average Power in the Specified Frequency Range
            Features(ii,1)=bandpower(Sections,SR,[aFreq(SpecifiedFreqRange) bFreq(SpecifiedFreqRange)]);
        end
            % Notification, User has not selected his/her sub-frequency/ies
    else
        error('ERROR, Please Select the Frequency Range(s)')
    end
end

        %% Normalization According to Min & Max   
% For us, that was not necessary to normalize the data
if (MinMaxNormalizationMethod==1)
        % Notification
    disp('Data Will Be Normalized')
    Min=min(Features);
    Max=max(Features);
    NormalizedPODF=[];
    for ij=1:length(Min)
        NormalizedPODF(:,ij)=(Features(:,ij)'-Min(ij))/(Max(ij)-Min(ij))*100;
    end
        % Replacing New Feature
    Features=[];Features=NormalizedPODF;
    clear NormalizedPODF Min Max
end

        %% Extracting Two Selected Stages
LabelsFeatures=[Labels,Features];
if (NonREMvsREM==1)                         % Selected Stages ---> NonREM & REM
    TwoClasses=[];TwoClasses=LabelsFeatures(find(Labels~=WakefulnessLabel),:);
    FirstStage='REM';SecondStage='NonREM';
    FirstClassLabel=REMLabel;SecondClassLabel=NonREMLabel;ExtraClassLabel=WakefulnessLabel;
        % Notification, Two selected stages
    disp(['Results will be published according to selected stages, ',SecondStage,' vs ',FirstStage])
elseif (NonREMvsWakefulness==1)             % Selected Stages ---> NonREM & Wakefulness
    TwoClasses=[];TwoClasses=LabelsFeatures(find(Labels~=REMLabel),:);
    FirstStage='Wakefulness';SecondStage='NonREM';
    FirstClassLabel=WakefulnessLabel;SecondClassLabel=NonREMLabel;ExtraClassLabel=REMLabel;
        % Notification, Two selected stages
    disp(['Results will be published according to selected stages, ',SecondStage,' vs ',FirstStage])
else                                         % Selected Stages ---> Wakefulness & REM
    TwoClasses=[];TwoClasses=LabelsFeatures(find(Labels~=NonREMLabel),:);
    FirstStage='Wakefulness';SecondStage='REM';
    FirstClassLabel=WakefulnessLabel;SecondClassLabel=REMLabel;ExtraClassLabel=NonREMLabel;
        % Notification, Two selected stages
    disp(['Results will be published according to selected stages, ',SecondStage,' vs ',FirstStage])
end

        %% Detecting and Replacing Outliers
Classes=[];
    % Class 1
Class1=[];Class1=TwoClasses(find(TwoClasses(:,1)==FirstClassLabel),:);
Class1=filloutliers(Class1(:,2:end),ReplacingMethod,DetectingMethod);
I=[];[I,~]=find(TwoClasses(:,1)==FirstClassLabel);
Classes=zeros(size(TwoClasses,1),size(TwoClasses,2));
Classes(I,1)=FirstClassLabel;Classes(I,2:end)=Class1;
    % Class 2
Class2=[];Class2=TwoClasses(find(TwoClasses(:,1)==SecondClassLabel),:);
Class2=filloutliers(Class2(:,2:end),ReplacingMethod,DetectingMethod);
I=[];[I,~]=find(TwoClasses(:,1)==SecondClassLabel);
Classes(I,1)=SecondClassLabel;Classes(I,2:end)=Class2;
TwoClasses=[];TwoClasses=Classes;
clear LabelsFeatures Classes

%%%%%%%%%%%%%%%%%%%%%%%  Classification
        %% Importing Model
LabelsPrim=[];LabelsPrim(:,1)=Labels;                                                   % All Labels

    % Notification, Number of iteration which user has been selected
disp([num2str(Iteration),' Time iteration starts'])

for Itr=1:Iteration
    
        % Shuffling Data
    % We shuffled data for achieving the results in different modes
    if (Itr~=1)
        ShuffleIndex=[];ShuffleIndex=randperm(size(TwoClasses,1));
        ShuffelFeatures=TwoClasses(ShuffleIndex,:);
            % Replacing features and related labels according to new index
        NewFeatures=[];NewFeatures=ShuffelFeatures(:,2:end);
        NewLabel=ShuffelFeatures(:,1);
    else
            % In the first loop
        NewFeatures=TwoClasses(:,2:end);
        NewLabel=TwoClasses(:,1);
    end
    
            %% Classification Method
        % Bayesian Classifier
    if (BayesianClassifierMethod==1)
        DataForAnalysing=[];DataForAnalysing=NewFeatures;
            % Loading imported model
        if (AveragePowerUsingPSD==1)
            Name=[];Name=(['KFoldKNBModelREMvsWakefulnessPSDMouse',num2str(MOUSE),'.mat']);
            load(Name);
        else
            if (NonREMvsREM==1 && NonREMvsWakefulness~=1)
                Name=[];Name=(['KFoldKNBModelNonREMvsREMMouse',num2str(MOUSE),'.mat']);
                load(Name);
            elseif (NonREMvsREM~=1 && NonREMvsWakefulness==1)
                Name=[];Name=(['KFoldKNBModelNonREMvsWakefulnessMouse',num2str(MOUSE),'.mat']);
                load(Name);
            elseif (NonREMvsREM~=1 && NonREMvsWakefulness~=1)                        % Default model (REM vs Wakefulness)
                Name=[];Name=(['KFoldKNBModelREMvsWakefulnessMouse',num2str(MOUSE),'.mat']);
                load(Name);
            end
        end
        Class = KernelNaiveBayesianKFoldModel.predictFcn(DataForAnalysing);
    end
    
    % Another models
    
        % Evaluation of First Stage
    GroupTEST=[];GroupTEST=NewLabel;
    [PrimaryAccuracy(Itr),PrimarySensitivity(Itr),PrimarySpecificity(Itr),PrimarySelectivity(Itr),PrimaryConfusionMatrix{Itr}]=...
        Evaluation(GroupTEST,Class,FirstClassLabel,SecondClassLabel);
    
            %% Verification (Hypnogram)
        % LabelsPrim: Column 1, All True Labels .... Column 2, Predicted Labels
     jj=1;
     for ii=1:length(Labels)
         if (Labels(ii,1)==ExtraClassLabel)
             LabelsPrim(ii,2)=ExtraClassLabel;
         else
             LabelsPrim(ii,2)=GroupTEST(jj,1);
             jj=jj+1;
         end
     end
    
        % Results in Each Iteration
     Hypnogram(Itr,:)=LabelsPrim(:,1);
     HypnogramPrim(Itr,:)=LabelsPrim(:,2);
     clear LabelsPrim
    
end

    % Total Results
MouseAccuracy=mean(PrimaryAccuracy);
MouseSensitivity=mean(PrimarySensitivity);
MouseSpecificity=mean(PrimarySpecificity);
MouseSelectivity=mean(PrimarySelectivity);
MouseConfusionMatrix{1,1}=PrimaryConfusionMatrix;
MouseDataForAnalysing{1,1}=DataForAnalysing;
MouseGroupTEST{1,1}=GroupTEST;
MouseClass{1,1}=Class;
MouseHypnogram{1,1}=Hypnogram;
MouseHypnogramPrim{1,1}=HypnogramPrim;

    % Notification, Results
if (AllFrequecyRanges==1)
    disp(['**************************          Total Results ----> ',SecondStage,' versus ', FirstStage,' of Mouse Number ',num2str(MOUSE),...
        'using all sub-frequencies'])
else
    disp(['**************************          Total Results ----> ',SecondStage,' versus ', FirstStage,' of Mouse Number ',num2str(MOUSE),...
        ' using single sub-frequency [ ',num2str(aFreq(SpecifiedFreqRange)),' - ',num2str(bFreq(SpecifiedFreqRange)),' ] Hz'])
end
disp(['Accuracy of ',SecondStage,' Stage =  ',num2str(MouseAccuracy(MOUSE,1)),' %']);
disp(['Sensitivity of ',SecondStage,' Stage =  ',num2str(MouseSensitivity(MOUSE,1)),' %']);
disp(['Specificity of ',SecondStage,' Stage =  ',num2str(MouseSpecificity(MOUSE,1)),' %']);
disp(['Selectivity of ',SecondStage,' Stage =  ',num2str(MouseSelectivity(MOUSE,1)),' %']);

        %% Saving Information
    % Parameters
MouseInformation(1).Parameters.MouseNumber=MOUSE;    
MouseInformation(1).Parameters.Sheet=Sheet;
MouseInformation(1).Parameters.Column=Column;
MouseInformation(1).Parameters.Threshold=Threshold;
MouseInformation(1).Parameters.Factor=Factor;
MouseInformation(1).Parameters.WindowRemoving=WindowRemoving;
MouseInformation(1).Parameters.TimeDuration=TimeDuration;
MouseInformation(1).Parameters.TimeSection=TimeSection;
MouseInformation(1).Parameters.WakefulnessLabel=WakefulnessLabel;
MouseInformation(1).Parameters.NonREMLabel=NonREMLabel;
MouseInformation(1).Parameters.REMLabel=REMLabel;
MouseInformation(1).Parameters.aFreq=aFreq;
MouseInformation(1).Parameters.bFreq=bFreq;
MouseInformation(1).Parameters.FirstStage=FirstStage;
MouseInformation(1).Parameters.SecondStage=SecondStage;
MouseInformation(1).Parameters.Iteration=Iteration;
MouseInformation(1).Parameters.DataSource=DataSource;
MouseInformation(1).Parameters.SamplingRate=SR;
MouseInformation(1).Parameters.AllFrequecyRanges=AllFrequecyRanges;
MouseInformation(1).Parameters.SpecifiedFreqRange=SpecifiedFreqRange;
    % Orders
MouseInformation(1).Orders.ArtifactRemoval=ArtifactRemoval;
MouseInformation(1).Orders.Normalization=MinMaxNormalizationMethod;
MouseInformation(1).Orders.ReplacingMethod=ReplacingMethod;
MouseInformation(1).Orders.DetectingMethod=DetectingMethod;
MouseInformation(1).Orders.AveragePowerUsingPSD=AveragePowerUsingPSD;
MouseInformation(1).Orders.NonREMvsREM=NonREMvsREM;
MouseInformation(1).Orders.NonREMvsWakefulness=NonREMvsWakefulness;
MouseInformation(1).Orders.BayesianClassifierMethod=BayesianClassifierMethod;
    % Results
MouseInformation(1).Results.Accuracy=MouseAccuracy;
MouseInformation(1).Results.Sensitivity=MouseSensitivity;
MouseInformation(1).Results.Specificity=MouseSpecificity;
MouseInformation(1).Results.Selectivity=MouseSelectivity;
MouseInformation(1).Results.ConfusionMatrix=MouseConfusionMatrix;
MouseInformation(1).Results.DataForAnalysing=MouseDataForAnalysing;
MouseInformation(1).Results.GroupTEST=MouseGroupTEST;
MouseInformation(1).Results.Class=MouseClass;
MouseInformation(1).Results.Hypnogram=MouseHypnogram;
MouseInformation(1).Results.HypnogramPrim=MouseHypnogramPrim;

    % Saving Results
FileSave=['TotalResults_Rahimi.et.al_',FirstStage,'vs',SecondStage,'MouseNo',num2str(MOUSE),'.mat'];
save(FileSave,'MouseInformation','-v7.3');

        %% Internal Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Accuracy,Sensitivity,Specificity,Selectivity,ConfusionMatrix]=Evaluation(NewScore,idx,FirstClassLabel,SecondClassLabel)

    % True Negative
I=[];I=find(NewScore==FirstClassLabel);
T=[];T=find(idx(I,:)==FirstClassLabel);
TN=length(T);                                        % Corecct Detection of First Stage
    % True Positive
I=[];I=find(NewScore==SecondClassLabel);
T=[];T=find(idx(I,:)==SecondClassLabel);
TP=length(T);                                        % Corecct Detection of Second Stage

    % False Positive
I=[];I=find(NewScore==FirstClassLabel);
T=[];T=find(idx(I,:)==SecondClassLabel);
FP=length(T);                                        % InCorecct Detection of First Stage
    % False Negative
I=[];I=find(NewScore==SecondClassLabel);
T=[];T=find(idx(I,:)==FirstClassLabel);
FN=length(T);                                        % InCorecct Detection of Second Stage

    % Accuracy
Accuracy=(TP+TN)/(TP+TN+FP+FN)*100;
Sensitivity=TP/(TP+FN)*100;
Specificity=TN/(TN+FP)*100;
Selectivity=TP/(TP+FP)*100;

    % Confusion Matrix
ConfusionMatrix(1,1)=TP/(TP+FN)*100;
ConfusionMatrix(2,2)=TN/(TN+FP)*100;
ConfusionMatrix(1,2)=FN/(FN+TP)*100;
ConfusionMatrix(2,1)=FP/(FP+TN)*100;
end
