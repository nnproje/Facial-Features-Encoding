#include <iostream>
#include "DataSet.h"
#include "NeuralNetwork.h"
#include <stdio.h>
#include "ConvFeedForward.h"
#define TRAIN_EXAMPLES 20000 //except for file number 8
#define ImageSize 40000
#define ImageDim 200
#define ReSize_Fact 1
#define Noise_Fact 0.1
#define Face_Size 100
#define Eyes_Size 20
#define Nose_Size 20
#define Mouth_Size 20
#define ENLARGE_FACT 0
#define Display_Data 1
#define Noisy_Data 0
#define LandMarks 0
#define Get_NewData 1
#define ever ;;
using namespace std;
int main()
{


	

	//------------------------------------------------------------------//
	//-------------------------- DataSet -------------------------------//
	//------------------------------------------------------------------//
	srand(time(NULL));
	clock_t START = clock();
	U_IntMatrix* X = new U_IntMatrix(TRAIN_EXAMPLES, ImageSize * (ReSize_Fact * ReSize_Fact));
	U_IntMatrix* Y = X;
	U_IntMatrix* noisyX = nullptr;
	U_IntMatrix* Faces = nullptr;
	U_IntMatrix* Eyes = nullptr;
	U_IntMatrix* Mouths = nullptr;
	U_IntMatrix* Noses = nullptr;
	const char* X_dir = "F:\\NEW_DATA\\1.txt";
	const char* NewData_dir = "F:\\NEW_DATA\\1.txt";
	const char* Noisy_dir = "F:\\Final Project\\DataSet\\NoisyImgs.txt";
	const char* ParametersPath = "F:\\Final Project\\DataSet\\Contractive+tiedNEW\\";
	get_dataset(X, Y, NewData_dir, X_dir, TRAIN_EXAMPLES, ImageSize, ImageDim, ReSize_Fact, Get_NewData);
	/*Shuffle(X, dummy);
	U_IntMatrix* X1 = X->SubMat(0, 0, X->Rows() - 1, 863);
	U_IntMatrix* X2 = X->SubMat(0, 864, X->Rows() - 1, -1);
	X1->WriteDataSet("F:\\Final Project\\DataSet\\XXX_TEST");
	X2->WriteDataSet("F:\\Final Project\\DataSet\\XXX_TRAIN");*/


	NoiseType NoiseT = SP;
	if (Noisy_Data)
	{
		if (NoiseT == Gauss)
			noisyX = AddGaussianNoise_Opencv(X, 200, 0.0, 0.1);
		else
			noisyX = add_SP_Noise(X, Noisy_dir, Noise_Fact, TRAIN_EXAMPLES);
	}
	
	if (LandMarks)
	{
		Faces = Face_Detection(X, ImageDim*ReSize_Fact, Face_Size); //please modify the path in the definition
		Faces->WriteDataSet("F:\\NEW_DATA\\1_FACES");
	}
	

	clock_t END = clock();

	/********************************************/
	if (Display_Data)
	{
		U_IntMatrix* src = new U_IntMatrix(ImageDim*ReSize_Fact, ImageDim*ReSize_Fact);
		for (int k = 0; k < X->Columns(); k++)
		{
			for (int n = 0; n < ImageDim*ReSize_Fact; n++)
				for (int m = 0; m < ImageDim*ReSize_Fact; m++)
				{
					src->access(n, m) = noisyX->access(n * ImageDim*ReSize_Fact + m, k);
				}
			visualize(ConvertMat_U(src, UC_F,NO));
		}
	}
	/********************************************/

	cout << endl << ">> DataSet Information:" <<endl;
	cout << "Training Images = " << (TRAIN_EXAMPLES* (ENLARGE_FACT + 1)) << endl;
	cout << "Preprocessing Time = " << (END - START) / CLOCKS_PER_SEC << " Secs " << endl <<endl;


	//------------------------------------------------------------------//
	//------------------- Network Architecture -------------------------//
	//------------------------------------------------------------------//
	int numOfLayers = 3;
	layer*  layers = new layer[numOfLayers];
	layers[0].put(ImageSize*ReSize_Fact*ReSize_Fact, NONE);
	layers[1].put(200, SIGMOID);
	layers[2].put(ImageSize*ReSize_Fact*ReSize_Fact, SIGMOID);
	float*  keep_prob = new float[numOfLayers];
	keep_prob[0] = 1;
	keep_prob[1] = 1;
	keep_prob[2] = 1;

	Arguments Arg;
	Arg.NetType = FC;
	Arg.layers = layers;
	Arg.numOfLayers = numOfLayers;
	Arg.X = X;
	Arg.Y = X;
	Arg.X_dev = X;
	Arg.Y_dev = X;
	Arg.X_test = X;
	Arg.Y_test = X;
	Arg.learingRate = 0.005;
	Arg.decayRate = 1;
	Arg.Rho = 0.05;
	Arg.beta_sparse = 0.1;
	Arg.numOfEpochs = 1;
	Arg.batchSize = 200;
	Arg.optimizer = ADAM;
	Arg.numPrint = 1;
	Arg.ErrType = CROSS_ENTROPY;
	Arg.regularizationParameter = 0;
	Arg.batchNorm = true;
	Arg.dropout = false;
	Arg.dropConnect = false;
	Arg.keep_prob = keep_prob;
	Arg.BatchMultiThread = false;
	Arg.EB = false;
	Arg.SPARSE = false;
	Arg.Stack = false;
	Arg.path = "Anything";
	Arg.Contractive = false;
	Arg.lambda_Contractive = 0.1;
	Arg.tiedWeights = false;
	if (Arg.Stack)
	{
		Arg.A = new Matrix(ImageSize, TRAIN_EXAMPLES);
		Arg.A->Read(X_dir);
	}


    //------------------------------------------------------------------//
    //---------------------- Print Network Layout ----------------------//
    //------------------------------------------------------------------//
    cout << ">> Training Information: " <<endl;
    cout << "Type Of Network: ";
    switch(Arg.NetType)
    {
        case FC: cout << "Fully Connected" << endl; break;
        case LENET1: cout << "LENET1" << endl; break;
        case CUSTOM: cout << "Convolution";
    }
    cout << "Optimization Algorithm: ";
    switch(Arg.optimizer)
    {
        case ADAM: cout << "ADAM" << endl; break;
        case GRADIENT_DESCENT: cout << "Gradient Descent" << endl;
    }
    cout << "Cost Function: ";
    switch(Arg.ErrType)
    {
        case CROSS_ENTROPY: cout<< "Cross Entropy" << endl; break;
        case SQAURE_ERROR: cout<< "Square Error" << endl;
    }
    cout << "Learining Rate = " << Arg.learingRate <<endl;
    cout << "Batch Size = " << Arg.batchSize <<endl;




    //------------------------------------------------------------------//
	//-------------------------- Training ------------------------------//
	//------------------------------------------------------------------//
    NeuralNetwork NN(&Arg);
	NN.RetrieveParameters(ParametersPath);
	int i = 0;
	for(;;)
    {
		
		//NN.test(TEST);
		//NN.Sparse_FeedForward(4);
        clock_t start = clock();
        cout << endl <<">> Epoch no. " << ++i << ":"<<endl;
        NN.train();
		if(i % 1 == 0 )
			NN.test(TEST);
        Arg.learingRate = Arg.learingRate * Arg.decayRate;
        clock_t end = clock();
        double duration_sec = double(end - start) / CLOCKS_PER_SEC;
        cout << "Time = " << duration_sec << endl;
		//NN.StoreParameters(ParametersPath);
    }
	_getche();
	return 0;
}