#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) 
	{
		VectorXd rmse(4);
		rmse << 0, 0, 0, 0;

		// check if there is a mismatch sizes between the estimations and the groundtruth
		//		 if the estimations size is not equal to zero		
		if (estimations.size() != ground_truth.size() || estimations.size() == 0)
		{
			cout << "Mismatch Sizes" << endl;
		}

		else
		{
			// accumulate squared residuals
			for (unsigned int i = 0; i < estimations.size(); ++i)
			{
				// Define the residual Vector
				VectorXd residual = estimations[i] - ground_truth[i];
				// Coefficient-wise multiplication
				residual = residual.array() * residual.array();
				rmse += residual;
			}

			// Calculate mean
			rmse = rmse / estimations.size();
			// Calculate the square root of the mean to have the root mean squared
			rmse = rmse.array().sqrt();
		}

		// return zeros in case of mismatch sizes or 
		// return the calculated RMSE between the estimations and the groundtruth 
		return rmse;
  
	}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) 
	{
		MatrixXd Hj(3, 4);
		// recover state parameters
		float px = x_state(0);
		float py = x_state(1);
		float vx = x_state(2);
		float vy = x_state(3);
		

		// Avoid Division by zero
		float Den = px*px + py*py; //C1
		if (Den < 0.0001)
		{
			Den = 0.0001;
		}
		// Compute Jacobian matrix elements
		float elem1				  = ((vx * py) - (vy * px));
		float SquareRootDen		  = sqrt(Den); //C2
		float DenXSquareRootDen	  = Den * SquareRootDen; //C3 
		float pxOverDen           = px / Den; //px / C1
		float pyOverDen			  = py / Den; // py / C1
		float pxOverSquareRootDen = px / SquareRootDen; // px / C2
		float pyOverSquareRootDen = py / SquareRootDen; // py / C2


		Hj <<	pxOverSquareRootDen,				pyOverSquareRootDen,					0,						0,
				-pyOverDen,							pxOverDen,								0,						0,
				py*elem1 / DenXSquareRootDen,		-px*elem1 / DenXSquareRootDen,	pxOverSquareRootDen, pyOverSquareRootDen;

		return Hj;
	}
