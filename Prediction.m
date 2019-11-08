mealData = importdata('MealNoMealData/mealData.csv');
mealData = flip(mealData,2);
mealArray = fillmissing(mealData,'nearest',2);
mealArray = fillmissing(mealArray,'nearest',1);
mealData = array2table(mealArray);

zeroes = [];
coefVariation = [];
top_ffts = [];
[n,w]=size(mealData);

for h = 1:n
    %Coefficient Variation
    rowMean = mean(mealData{h,1:w});
    rowStdDev = std(mealData{h,1:w});
    coefVariation = [coefVariation; rowStdDev/rowMean];

    %Zero crossing with CGM Velocity
    cgmVelocity = mealData{h,1:w-1} - mealData{h,2:w}; 
    [G,H] = find(cgmVelocity < 0.5);
    zeroes = [zeroes; H(end)];

    %Fast Fourier Transform
    fft_mealData(h,[1:w])= abs(fft(mealArray(h,[1:w])));  
    top_ffts(h,[1:8]) = fft_mealData(h,[2:9]);
end

%Root Mean Squared
RMS = rms(mealArray,2);

%Feature Matrix
FeatureMatrixMeal = [zeroes RMS coefVariation top_ffts];

%Normalizing the feature matrix between range of 0 to 1
FeatureMatrixMeal = normalize(FeatureMatrixMeal, 'range');

%Performing PCA
[coeff, score, latent] = pca(FeatureMatrixMeal);
our_top_eigens = coeff(:,1:5);

%Updated feature matrix for meal data
new_features_meal = FeatureMatrixMeal * our_top_eigens;

[r,c] = size(new_features_meal);
labelMeal = ones(r,1);

plot(new_features_meal);
xlabel('Day'), ylabel('Value');
title('Top 5 PCA Features Per Day For Patient 1 MEAL data');
legend({'Feature 1','Feature 2','Feature 3','Feature 4','Feature 5'},'Location','northeast');

%%%% No meal calculations

Nomeal = importdata('MealNoMealData/Nomeal.csv');
Nomeal = flip(Nomeal.data,2);
NomealArray = fillmissing(Nomeal,'nearest',2);
NomealArray = fillmissing(NomealArray,'nearest',1);
Nomeal = array2table(NomealArray);

zeroes = [];
coefVariation = [];
top_ffts = [];
[n,w]=size(Nomeal);

for h = 1:n
    %Coefficient Variation
    rowMean = mean(Nomeal{h,1:w});
    rowStdDev = std(Nomeal{h,1:w});
    coefVariation = [coefVariation; rowStdDev/rowMean];

    %Zero crossing with CGM Velocity
    cgmVelocity = Nomeal{h,1:w-1} - Nomeal{h,2:w}; 
    [G,H] = find(cgmVelocity < 0.5);
    zeroes = [zeroes; H(end)];

    %Fast Fourier Transform
    fft_Nomeal(h,[1:w])= abs(fft(NomealArray(h,[1:w])));  
    top_ffts(h,[1:8]) = fft_Nomeal(h,[2:9]);
end

%Root Mean Squared
RMS = rms(NomealArray,2);

%Feature Matrix
FeatureMatrixNoMeal = [zeroes RMS coefVariation top_ffts];

%Normalizing the feature matrix between range of 0 to 1
FeatureMatrixNoMeal = normalize(FeatureMatrixNoMeal, 'range');

%Updated feature matrix for no meal data
new_features_no_meal = FeatureMatrixNoMeal * our_top_eigens;

%Labels for no meal data
[r1,c1] =size(new_features_no_meal);
labelNoMeal = zeros(r1,1);

figure()
plot(new_features_no_meal);
xlabel('Day'), ylabel('Value');
title('Top 5 PCA Features Per Day For Patient 1 NO MEAL data');
legend({'Feature 1','Feature 2','Feature 3','Feature 4','Feature 5'},'Location','northeast');

% Create Training Data and Labels Data
TrainingData = [new_features_meal;new_features_no_meal];
LabelsData = [labelMeal;labelNoMeal];

% k fold cross validation
num_Folds=10;
Indices=crossvalind('Kfold',LabelsData,num_Folds);
for i=1:num_Folds
    TestFoldSamples=TrainingData(Indices==i,:);
    TrainFoldSamples=TrainingData(Indices~=i,:);
    TrainFoldLabel=LabelsData(Indices~=i,:);
    TestFoldLabel=LabelsData(Indices==i,:);
    
    % Train & Test linear SVM
    linearSVM = fitcsvm(TrainFoldSamples,TrainFoldLabel);
    predict_linearSVM=predict(linearSVM,TestFoldSamples);
    
    % Train & Test polynomial SVM
    polynomialSVM = fitcsvm(TrainFoldSamples,TrainFoldLabel, 'KernelFunction', 'polynomial');
    predict_polynomialSVM=predict(polynomialSVM,TestFoldSamples);
    
    %Train & Test Gaussian SVM
    gaussianSVM = fitcsvm(TrainFoldSamples,TrainFoldLabel,'KernelFunction','gaussian');
    predict_gaussianSVM=predict(gaussianSVM,TestFoldSamples);
    
    % Train & Test Naive Bayes
    naiveBayes = fitcnb(TrainFoldSamples,TrainFoldLabel);
    predict_naiveBayes=predict(naiveBayes,TestFoldSamples);
    
    % Calculating accuracy measures for linear SVM, polynomial SVM,
    % Gaussian SVM & Naive Bayes
    linearSVM_accuracy(i,1)=sum(grp2idx(predict_linearSVM)==grp2idx(TestFoldLabel))/length(TestFoldLabel);
    polynomialSVM_accuracy(i,1)=sum(grp2idx(predict_polynomialSVM)==grp2idx(TestFoldLabel))/length(TestFoldLabel);
    gaussianSVM_accuracy(i,1)=sum(grp2idx(predict_gaussianSVM)==grp2idx(TestFoldLabel))/length(TestFoldLabel);
    naiveBayes_accuracy(i,1)=sum(grp2idx(predict_naiveBayes)==grp2idx(TestFoldLabel))/length(TestFoldLabel);
    
end
save('polynomialSVM.mat','polynomialSVM','our_top_eigens');
save('linearSVM.mat','linearSVM','our_top_eigens');
save('naiveBayes.mat','naiveBayes','our_top_eigens');
save('gaussianSVM.mat','gaussianSVM','our_top_eigens');

disp("Average linear SVM Accuracy: " +mean(linearSVM_accuracy));
disp("Average polynomial SVM Accuracy: " +mean(polynomialSVM_accuracy));
disp("Average Gaussian SVM Accuracy: " +mean(gaussianSVM_accuracy));
disp("Average Naive Bayes Accuracy: " +mean(naiveBayes_accuracy));
%testScript("C:\Users\rahul reddy\Documents\RAHUL\CSE572Assignment2\MealNoMealData OLD\mealData1.csv");