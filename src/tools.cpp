#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;


const double Tools::epsilon = std::numeric_limits<float>::epsilon();
//const double Tools::epsilon = 1.0e-5;


VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	const auto nCount = estimations.size();

	assert(nCount > 0);
	assert(nCount == ground_truth.size());
	if (!nCount || (nCount != ground_truth.size())) {
		std::cout << "Error: vectors have different size!" << std::endl;
		return rmse;
	}

	VectorXd res(4);

	for (size_t i = 0; i < nCount; i++) {
		res = estimations[i] - ground_truth[i];
		res = res.array() * res.array();        
		rmse += res;
	}

	rmse /= static_cast<double>(nCount);
	rmse = rmse.array().sqrt();
	return rmse;
}
