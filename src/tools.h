#ifndef TOOLS_H_
#define TOOLS_H_

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits> 
#include "Eigen/Dense"

namespace Tools
{
	extern const double epsilon;

	// A helper method to calculate RMSE.
	Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations,
                                  const std::vector<Eigen::VectorXd> &ground_truth);

	// Returns projected values to X and Y as 2D vector.
	Eigen::VectorXd PolarToCartesian(double rho, double phi);
    
    double NormalizeAngle1(double phi);
    double NormalizeAngle(double phi);
};


inline Eigen::VectorXd Tools::PolarToCartesian(double rho, double phi)
{
	Eigen::Vector2d result;
	result << rho * cos(phi), rho * sin(phi);
	return result;
}

inline double Tools::NormalizeAngle(double phi)
{
    static const double pi2 = 2. * M_PI;
    
    if (phi < -M_PI || phi > M_PI )
    {
        phi = std::fmod(phi, pi2);
        if (phi > M_PI) phi -= pi2;
        if (phi < -M_PI) phi += pi2;
    }
    return phi;
}

#endif /* TOOLS_H_ */
