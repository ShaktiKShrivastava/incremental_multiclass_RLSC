
clc;
close all;
confIncremental;

%% Experimental setup
run_inc_rlsc_yesrec = 1;    % Incremental RLSC with recoding
computeTestAcc = 1;         % Flag for test accuracy computation
trainPart = 0.8;            % Training set part (1 - validation_percentage)
maxiter = 300;              % Maximum number of updates
numrep = 2;                 % Number of repetitions of the experiment
saveResult = 1;             % Save final results flag

switch datasetName
    case 'MNIST'
        dataConf_MNIST_inc;
    otherwise
        error('Dataset not recognized.')
end

if strcmp(coding, 'zeroOne') ~= 1
    error('This script uses the recoding type of the form: X''*Y*C. It is only compatible with the zeroOne coding.')
end

%% Tikhonov regularization hyperparameter (lambda) range

numLambdas = 20;    % Number of guesses
minLambdaExp = -3;  % Exponent of the minimum guess
maxLambdaExp = 0;   % Exponent of the maximum guess
lrng = logspace(maxLambdaExp , minLambdaExp , numLambdas);  % Lambda guesses array

%lambda varies between 0 to 1 as per the paper also
%lrng is 20 equispaced values between 0.001 = pow(10,-3) and 1 = pow(10,0)


%% Instantiate results storage structure
%numrep is the number of repetitions of the experiment
results = repmat(struct(...
            'testCM' , zeros(numrep,numel(classes),numSnaps, numel(classes), numel(classes)),...
            'bestValAccBuf' , zeros(numrep,numel(classes),numSnaps),...
            'bestCMBuf' , zeros(numrep,numel(classes),numSnaps, numel(classes), numel(classes)),...
            'bestLambdaBuf' , zeros(numrep,numel(classes),numSnaps),...
            'valAcc' , zeros(numrep,numel(classes),numSnaps,numLambdas),...
            'teAcc' , zeros(numrep,numel(classes),numSnaps,numLambdas),...
            'trainTime' , zeros(numrep,numel(classes),numSnaps),...
            'testAccBuf' , zeros(numrep,numel(classes),numSnaps)...
            ), ...
            numAlpha, 1);
    
for k = 1:numrep
    
    clc
    display(['Repetition # ', num2str(k), ' of ' , num2str(numrep)]);
    display(' ');
    display(['alpha recoding parameter values: ', num2str(alphaArr)]);
    display(' ');
    
    %fancy line for no reason
    progressBar(k,numrep);
    display(' ');
    display(' ');

    %% Load dataset

    ds = dsRef(ntr , nte, coding , 0, 0, 0, {classes , trainClassFreq, testClassFreq});

    % Mix up sampled points
    ds.mixUpTrainIdx;
    ds.mixUpTestIdx;
  
    Xtr = ds.X(ds.trainIdx,:);
    Xte = ds.X(ds.testIdx,:);
    Ytr = ds.Y(ds.trainIdx,:);
    Yte = ds.Y(ds.testIdx,:);
    %display(ds.trainIdx)
    
    %display(Xtr);
    %display(Xte);
    %display(Ytr);
    %display(Yte);
    
    
    ntr = size(Xtr,1);
    nte = size(Xte,1);
    d = size(Xtr,2);
    t  = size(Ytr,2);
    p = ds.trainClassNum / ntr; % Class frequencies array
    
    
    switch datasetName
        case 'MNIST'
            
            %TODO
            %find out why splitting is needed here
            %its needed to do the training and the validation
            % Splitting MNIST
            
            ntr1 = round(ntr*trainPart);
            nval1 = round(ntr*(1-trainPart));
            tr1idx = 1:ntr1;
            val1idx = (1:nval1) + ntr1;
            Xtr1 = Xtr(tr1idx,:);
            Xval1 = Xtr(val1idx,:);
            Ytr1 = Ytr(tr1idx,:);
            Yval1 = Ytr(val1idx,:);
        otherwise
            error('Dataset not recognized.')
    end
    
       
    % Cycle over the imbalanced class array
    % At each cycle, a different class (specified in 'imbClassArr') is unbalanced, while all the
    % others are kept balanced.
    for imbClass = imbClassArr
        
        display(['Imbalanced class: ', num2str(imbClass)])
        
        % Split training set in balanced (for pretraining) and imbalanced
        % (for incremental learning) subsets
        
        %this line is a stunt, see the notebook for the explanation
        
        
        
        %debugging line
        %Ytr1 = Ytr1(1000:2000,:);
        
        
        [tmp1,tmp2] = find(Ytr1 == 1);
        
        
        
        %we will have to tweak the values of all these parameters,
        %specially idx_bal and idx_imbal
        %okay, so now these few lines are only computing which and all
        %points are having balanced sample, and which not
        idx_bal = tmp1(tmp2 ~= imbClass);   % Compute indexes of balanced samples
        Xtr_bal = Xtr1(idx_bal , :);
        Ytr_bal = Ytr1(idx_bal , :);
        ntr_bal = size(Xtr_bal,1);  %which is same as size(idx_bal,1)
        
        idx_imbal = setdiff(1:ntr1 , idx_bal);   % Compute indexes of imbalanced samples
        Xtr_imbal = Xtr1(idx_imbal , :);
        Ytr_imbal = Ytr1(idx_imbal , :);
        ntr_imbal = min([maxiter, numel(idx_imbal)]);
        %finding points over
        
        
        % Pre-train batch model only on points belonging balanced classes
        XtX = Xtr_bal'*Xtr_bal;     %XtX is (Xtr_bal)transpose * (Xtr_bal)
        XtY = Xtr_bal'*Ytr_bal;     %XtY is (Xtr_bal)transpose * (Ytr_bal)

        lstar = 0;      % Best lambda
        bestAcc = 0;    % Highest accuracy
        w = cell(1,numel(lrng));
        R = cell(1,numel(lrng));
        
        %what is this cholesky matrix doing? Some upper traingular matrix
        %these guys are trying to find out
        for lidx = 1:numel(lrng)
            l = lrng(lidx);
            R{lidx} = chol(XtX + ntr_bal * l * eye(d), 'upper');  
        end
        
        
                
        %% Incremental RLSC, with recoding
        % Incremental (or Recursive) Regularized Least Squares for Classification, 
        % with Tikhonov regularization parameter selection

        if run_inc_rlsc_yesrec == 1    

            %Init
            Xtr_tmp = zeros(size(Xtr_bal,1)+size(Xtr_imbal,1),d);
            Ytr_tmp = zeros(size(Ytr_bal,1)+size(Ytr_imbal,1),t);

            Xtr_tmp(1:ntr_bal,:) = Xtr_bal;
            Ytr_tmp(1:ntr_bal,:) = Ytr_bal;

            R_tmp = cell(1,numLambdas);
            trainTime = 0;
            ntr_tmp = size(Xtr_bal,1);

            sIdx = 1;
            % cycle over the imbalanced class samples
            
            %so from here I am assuming that we can refer Xtr_tmp and
            %Ytr_tmp to the training data from our fingerprints, i.e. I am
            %trying to remove the need of all the previous variables that
            %were accessing the database of images, and assume that this is
            %the first occurence where the substitution can be made
            for q = 1:ntr_imbal

                
                ntr_tmp = ntr_tmp + 1;
                
                %replace Xtr_imbal and Ytr_imbal in the code
                Xtr_tmp(ntr_tmp,:) = Xtr_imbal(q,:);
                Ytr_tmp(ntr_tmp,:) = Ytr_imbal(q,:);
                
                
                tic

                % Compute p
                % p: Relative class frequencies vector
                [~,tmp] = find(Ytr_tmp == 1);
                a = unique(tmp);
                out = [a,histc(tmp(:),a)];
                
                %number of data points of each class divided by the number
                %of elements of each class (not the predicted labels,
                %rather the already labelled ones
                p = out(:,2)'/ntr_tmp;

                % Compute t x t recoding matrix C
                C = zeros(t);
                for i = 1:t
                    currClassIdx = i;
                    C(i,i) = computeGamma(p,currClassIdx);
                end

                % Compute b
                XtY_tmp = Xtr_tmp(1:ntr_tmp,:)' * Ytr_tmp(1:ntr_tmp,:);
                %what is the role of XtY_tmp???
                
                % Buffer variables
                lstar = zeros(1,numAlpha);         % Best lambda
                wstar = zeros(d,t,numAlpha);       % Best w
                currAcc = zeros(1,numAlpha);       % Current accuracy
                bestAcc = zeros(1,numAlpha);       % Highest accuracy
                CM = zeros(t,t,numAlpha);          % coding matrix temporary
                bestCM = zeros(t,t,numAlpha);      %Best coding matrix seen till now
                
                for lidx = 1:numel(lrng)

                    l = lrng(lidx);

                    if q == 1
                        % Compute first Cholesky factorization of XtX + n * lambda * I
                        R_tmp{lidx} = R{lidx};  
                    end
                    
                    % Update Cholesky factor R
                    R_tmp{lidx} = cholupdatek(R_tmp{lidx}, Xtr_imbal(q,:)' , '+');                

                    
                    %sIdx seems to be a variable used only for extra
                    %caution, because sIdx is serving the same purpose as q
                    %it seems
                    if (sIdx <= numel(snaps)) && (q == snaps(sIdx))

                        w0 = R_tmp{lidx} \ (R_tmp{lidx}' \ XtY_tmp);                    

                        for kk = 1:numAlpha

                            alpha = alphaArr(kk);       
                            
                            % Training with specified alpha
                            w = w0 * (C ^ alpha);

                            % Predict validation labels
                            Yval1pred_raw = Xval1 * w;
                            %So from here they'll try finding out the best
                            %value of w for which the Yval1pred_raw is as
                            %close as possible to the real values, so bro,
                            %this is training. w depended on lambda and
                            %alpha, and here these iterations will be done
                            %for 2 values of alpha and 20 values of lambda,
                            %so yes, this will find out the best value of w
                            %for sure.
                            
                            % Compute current validation accuracy
                            if t > 2
                                Yval1pred = scoresToClasses( Yval1pred_raw , coding );
                                %checking if the validation to classes is
                                %accessing anything to of Xval or not
                                [currAcc(kk) , CM(:,:,kk)] = weightedAccuracy2( Yval1, Yval1pred , classes);
                                %so at the last line they've found out some
                                %by some magic, the current accuracy
                                
                                
                            else
                                CM(:,:,kk) = confusionmat(Yval1,sign(Yval1pred_raw));
                                CM(:,:,kk) = CM(:,:,kk) ./ repmat(sum(CM(:,:,kk),2),1,2);
                                currAcc(kk) = trace(CM(:,:,kk))/2;                
                            end

                            results(kk).valAcc(k,imbClass,sIdx,lidx) = currAcc(kk);

                            if currAcc(kk) > bestAcc(kk)
                                bestAcc(kk) = currAcc(kk);
                                bestCM(:,:,kk) = CM(:,:,kk);
                                lstar(kk) = l;
                                wstar(:,:,kk) = w;
                            end

                            results(kk).ntr = ntr;
                            results(kk).nte = nte;
                            results(kk).bestValAccBuf(k,imbClass,sIdx) = bestAcc(kk);
                            results(kk).bestCMBuf(k,imbClass,sIdx,:,:) = bestCM(:,:,kk);
                            results(kk).bestLambdaBuf(k,imbClass,sIdx) = lstar(kk);     
                            
                        end
                    end
                end
                
                % Compute test accuracy
                if (computeTestAcc == 1) && (sIdx <= numel(snaps)) && (q == snaps(sIdx))
                    for kk = 1:numAlpha

                        % Predict test labels
                        Ytepred_raw = Xte * wstar(:,:,kk);

                        % Compute current test accuracy

                        if t > 2
                            Ytepred = scoresToClasses( Ytepred_raw , coding );
                            [teAcc , CM] = weightedAccuracy2( Yte, Ytepred , classes);
                        else
                            CM = confusionmat(Yte,sign(Ytepred_raw));
                            CM = CM ./ repmat(sum(CM,2),1,2);
                            teAcc = trace(CM)/2;
                        end        

                        % Save test accuracy and confusion matrix (CM)
                        results(kk).testAccBuf(k,imbClass,sIdx) = teAcc;
                        results(kk).testCM(k,imbClass,sIdx,:,:) = CM;
                    end
                end
                
                % Update snapshot index
                if (sIdx < numel(snaps)) && (q == snaps(sIdx))
                    sIdx = sIdx + 1;
                end
            end
        end
    end
    

    %% Update saved workspace at each repetition

    if saveResult == 1

        save([resdir '/workspace.mat'] , '-v7.3');
    end    
    
end



%% Plots (class by class)

for c = imbClassArr
    
    % Test error comparison plots
    
    if numrep == 1
        warning('Plots only for numrep > 1');
    else
        % Overall Test Accuracy
        
        c2 = squeeze(results(recod_alpha_idx).testAccBuf(:,c,:));
        c3 = squeeze(results(1).testAccBuf(:,c,:));
        
        m_rec_tot_acc_te = mean(c2,1);
        s_rec_tot_acc_te = std(c2,[],1);
        
        m_nai_tot_acc_te = mean(c3,1);
        s_nai_tot_acc_te = std(c3,[],1);
        
        
        for kk = 2: numAlpha
            
            figure
                        
            box on
            grid on
            hold on
            
            h1 = bandplot(snaps,c3, ...
                'r' , 0.1 , 0 , 1 , '-');
            h2 = bandplot(snaps,c2, ...
                'b' , 0.1 , 0 , 1 , '-');
            xlabel('n_{imb}','FontSize',16)
            ylabel('Overall Test Accuracy','FontSize',16)
            title(['Imbalanced class: ' , num2str(c), ' of ' , num2str(t)])
            hold off    
        end        
        
        %%% Imbalanced Test Accuracy
        
        c2 = squeeze(results(recod_alpha_idx).testCM(:,c,:, c, c));
        c3 = squeeze(results(1).testCM(:,c,:, c, c));
        

        m_rec_imb_acc_te = mean(c2,1);
        s_rec_imb_acc_te = std(c2,[],1);
            
        m_nai_imb_acc_te = mean(c3,1);
        s_nai_imb_acc_te = std(c3,[],1);
            
        % C = 28, separate figures for accuracy section
        for kk = 2: numAlpha
            
            figure
                        
            box on
            grid on
            hold on
            
            h1 = bandplot(snaps,c3, ...
                'r' , 0.1 , 0 , 1 , '-');
            h2 = bandplot(snaps,c2, ...
                'b' , 0.1 , 0 , 1 , '-');
            xlabel('n_{imb}','FontSize',16)
            ylabel('Imbalanced Test Accuracy','FontSize',16)
            title(['Imbalanced class: ' , num2str(c), ' of ' , num2str(t)])
            hold off    
        end                
        
        
        %%% Balanced Test Accuracy

        a2 = squeeze(results(recod_alpha_idx).testAccBuf(:,c,:));
        b2 = squeeze(results(recod_alpha_idx).testCM(:,c,:, c, c));
        c2 = (t*a2 - b2) / (t-1);

        a3 = squeeze(results(1).testAccBuf(:,c,:));
        b3 = squeeze(results(1).testCM(:,c,:, c, c));
        c3 = (t*a3 - b3) / (t-1);

        m_rec_bal_acc_te = mean(c2,1);
        s_rec_bal_acc_te = std(c2,[],1);
            
        m_nai_bal_acc_te = mean(c3,1);
        s_nai_bal_acc_te = std(c3,[],1);
            
            
        % C != 28, separate figures for accuracy section
        for kk = 2: numAlpha
            
            figure
                        
            box on
            grid on
            hold on
            
            h1 = bandplot(snaps,c3, ...
                'r' , 0.1 , 0 , 1 , '-');
            h2 = bandplot(snaps,c2, ...
                'b' , 0.1 , 0 , 1 , '-');
            xlabel('n_{imb}','FontSize',16)
            ylabel('Balanced Test Accuracy','FontSize',16)
            title(['Imbalanced class: ' , num2str(c), ' of ' , num2str(t)])
            hold off    
        end      
        
    end    
end


%% Plots (averaged over classes first, and then on repetitions)

% Test error comparison plots

if numrep == 1
    warning('Plots only for numrep > 1');
else

    % Overall Test Accuracy

    % mean on class first, then on rep
    c2 = squeeze(mean(results(recod_alpha_idx).testAccBuf(:,:,:),2));
    c3 = squeeze(mean(results(1).testAccBuf(:,:,:),2));

    m_rec_tot_acc_te = mean(c2,1);
    s_rec_tot_acc_te = std(c2,[],1);

    m_nai_tot_acc_te = mean(c3,1);
    s_nai_tot_acc_te = std(c3,[],1);

    for kk = 2: numAlpha

        figure

        box on
        grid on
        hold on

        h1 = bandplot(snaps,c3, ...
            'r' , 0.1 , 0 , 1 , '-');
        h2 = bandplot(snaps,c2, ...
            'b' , 0.1 , 0 , 1 , '-');
        xlabel('n_{imb}','FontSize',16)
        ylabel('Overall Test Accuracy','FontSize',16)
        hold off    
    end


    %%% Imbalanced Test Accuracy

    c2=0;
    c3=0;

    for c_idx = 1:numel(imbClassArr)
        c = imbClassArr(c_idx);

        c2 = c2 + squeeze(results(recod_alpha_idx).testCM(:, c, :, c, c)) / numel(imbClassArr);
        c3 = c3 + squeeze(results(1).testCM(:, c, :, c, c)) / numel(imbClassArr);
    end

    m_rec_imb_acc_te = mean(c2,1);
    s_rec_imb_acc_te = std(c2,[],1);

    m_nai_imb_acc_te = mean(c3,1);
    s_nai_imb_acc_te = std(c3,[],1);

    % C = 28, separate figures for accuracy section
    for kk = 2: numAlpha

        figure

        box on
        grid on
        hold on

        h1 = bandplot(snaps,c3, ...
            'r' , 0.1 , 0 , 1 , '-');
        h2 = bandplot(snaps,c2, ...
            'b' , 0.1 , 0 , 1 , '-');
        xlabel('n_{imb}','FontSize',16)
        ylabel('Imbalanced Test Accuracy','FontSize',16)
        hold off    
    end                


    %%% Balanced Test Accuracy

    a2=0;
    b3=0;
    c2=0;
    a3=0;
    b2=0;
    c3=0;

    for c_idx = 1:numel(imbClassArr)
        c = imbClassArr(c_idx);

        a2 = squeeze(results(recod_alpha_idx).testAccBuf(:,c,:));
        b2 = squeeze(results(recod_alpha_idx).testCM(:,c,:, c, c));
        c2 = c2 + ((t*a2 - b2) / (t-1)) / numel(imbClassArr);

        a3 = squeeze(results(1).testAccBuf(:,c,:));
        b3 = squeeze(results(1).testCM(:,c,:, c, c));
        c3 = c3 + ((t*a3 - b3) / (t-1)) / numel(imbClassArr);
    end        

    m_rec_bal_acc_te = mean(c2,1);
    s_rec_bal_acc_te = std(c2,[],1);

    m_nai_bal_acc_te = mean(c3,1);
    s_nai_bal_acc_te = std(c3,[],1);


    % C != 28, separate figures for accuracy section
    for kk = 2: numAlpha

        figure

        box on
        grid on
        hold on

        h1 = bandplot(snaps,c3, ...
            'r' , 0.1 , 0 , 1 , '-');
        h2 = bandplot(snaps,c2, ...
            'b' , 0.1 , 0 , 1 , '-');
        xlabel('n_{imb}','FontSize',16)
        ylabel('Balanced Test Accuracy','FontSize',16)
        hold off    
    end      

end

%% Save figures

figsdir = [ resdir , '/figures/'];
mkdir(figsdir);
saveAllFigs;


%% Save workspace

if saveResult == 1

    save([resdir '/workspace.mat'] , '-v7.3');
end
