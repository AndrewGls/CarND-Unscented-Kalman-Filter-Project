#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
: is_initialized_(false)
, previous_timestamp_(0)
, n_x_(5)
, n_aug_(7)         // Adds noise std_a & std_yawdd_ to the motion model in Predict state.
, n_sigma_(2*n_aug_ + 1)
, n_z_lidar_(2)
, n_z_radar_(3)
, n_std_a_(5)
, n_std_yawdd_(6)
, lambda_(3 - n_aug_)
{
    // if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

    // initial state vector
    x_ = VectorXd(n_x_);
    x_.fill(0);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    
    // predicted sigma points matrix
    Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3.; // tune the noise value!!!
    
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.7;//M_PI / 2.; // tune the noise value!!!
    
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    R_laser_ = MatrixXd(n_z_lidar_, n_z_lidar_);
    R_laser_ << std_laspx_ * std_laspx_, 0,
                0, std_laspy_ * std_laspy_;
    
    R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
    R_radar_ << std_radr_ * std_radr_, 0, 0,
                0, std_radphi_ * std_radphi_, 0,
                0, 0, std_radrd_ * std_radrd_;
    
    Xsig_aug_ = MatrixXd(n_aug_, n_sigma_);
    
    //create vector for weights
    weights_ = VectorXd(n_sigma_);
}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& pack)
{
    if (!is_initialized_) {
        Init(pack);
        return;
    }
    
    // Time is measured in seconds.
    double dt = (pack.timestamp_ - previous_timestamp_) / 1.e6;
    assert(dt >= 0.);
    
    // Do prediction.
	// Division large time step into small steps helps to maintain numerical stability.
	// Without this step the UKF generates larger RMSE. To reduce RMSE it is possible to reduce std_a_ & std_yawdd_.
    while (dt > 0.1) {
        Prediction(0.05);
        dt -= 0.05;
    }
    
    Prediction(dt);
    
    // Do Update.
    if (use_laser_ && pack.IsLidar()) {
        UpdateLidar(pack);
        previous_timestamp_ = pack.timestamp_;
    } else if (use_radar_ && pack.IsRadar()) {
        UpdateRadar(pack);
        previous_timestamp_ = pack.timestamp_;
    }
}

void UKF::Init(const MeasurementPackage& pack)
{
    assert(!is_initialized_);
    
    // Init model covariance matrix
    P_.fill(0);
    for (int i = 0; i < P_.cols(); i++) {
        P_(i,i) = 1;
    }
    
    //set weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < n_sigma_; i++)
        weights_(i) = 0.5 / (lambda_ + n_aug_);
    
    previous_timestamp_ = pack.timestamp_;
    
    const auto& z = pack.raw_measurements_;
    
    if (pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Convert radar from polar to cartesian coordinates and initialize state as (px, py, v, yaw_angle, yaw_rate)
        const auto rho = z(0);
        const auto phi = z(1);
        const auto& pos = Tools::PolarToCartesian(rho, phi); // returns (px, py)
        x_ << pos(0), pos(1), 0, 0, 0;                    // sets (px, py, v, yaw_angle, yaw_rate)
    }
    else if (pack.sensor_type_ == MeasurementPackage::LASER) {
        // Initialize state as (px, py, v, yaw_angle, yaw_rate)
        x_ << z(0), z(1), 0, 0, 0;
    }
    
    is_initialized_ = true;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
    // Estimate the object's location. Modify the state
    // vector, x_. Predict sigma points, the state, and the state covariance matrix.
    CalcAugmentedSigmaPoints();
    PredictSigmaPoints(delta_t);
    // predict the state mean vector
    x_ = UKF::CalcPredictedMean(Xsig_pred_, weights_);
	// predict the state covariance matrix
    PredictCovariance();    
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage& pack)
{
    // Predict measurement:
    
    // predicted measurement sigma points matrix by transforming sigma points into measurement space
    MatrixXd Zsig_pred = TransformLidarSigmaPointsIntoMeasurementSpace();
    
    // calculate mean predicted measurement
    VectorXd z_pred = UKF::CalcPredictedMean(Zsig_pred, weights_);
    assert(z_pred.rows() == n_z_lidar_);
    
    // Calculate measurement covariance matrix S and cross correlation Tc matrix:
    
    // measurement covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z_lidar_, n_z_lidar_);
    // cross correlation Tc matrix
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_lidar_);
    
    for (int i = 0; i < n_sigma_; i++)
    {
        //residual
        VectorXd z_diff = Zsig_pred.col(i) - z_pred;
        
        S += weights_(i) * z_diff * z_diff.transpose();
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;        
        x_diff(3) = Tools::NormalizeAngle(x_diff(3));
        
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    
    //add measurement noise covariance matrix
    S += R_laser_;
    
    
    // Update measurement state:
    
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    const VectorXd& z = pack.raw_measurements_;
    assert(z.rows() == 2);
    
    //residual
    VectorXd z_diff = z - z_pred;
    
    // Update state mean and covariance matrix
    x_ += K * z_diff;
    x_(3) = Tools::NormalizeAngle(x_(3));
    P_ -= K * S * K.transpose();
    
    // Calculate Normalized Innovation Squared (NIS).
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& pack)
{
    // Predict measurement:
    
    // predicted measurement sigma points matrix by transforming sigma points into measurement space.
    MatrixXd Zsig_pred = TransformRadarSigmaPointsIntoMeasurementSpace();
    
    // calculate mean predicted measurement
    VectorXd z_pred = UKF::CalcPredictedMean(Zsig_pred, weights_);
    assert(z_pred.rows() == n_z_radar_);
    
    // Calculate measurement covariance matrix S and cross correlation Tc matrix:
    
    // measurement covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);
    
    // cross correlation Tc matrix
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);
    
    for (int i = 0; i < n_sigma_; i++)
    {
        //residual
        VectorXd z_diff = Zsig_pred.col(i) - z_pred;        
        z_diff(1) = Tools::NormalizeAngle(z_diff(1));
        
        S += weights_(i) * z_diff * z_diff.transpose();
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;        
        x_diff(3) = Tools::NormalizeAngle(x_diff(3));
        
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    
    //add measurement noise covariance matrix
    S += R_radar_;

    
    // Update measurement state:
    
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    const VectorXd& z = pack.raw_measurements_;
    assert(z.rows() == 3);
    
    //residual
    VectorXd z_diff = z - z_pred;    
    z_diff(1) = Tools::NormalizeAngle(z_diff(1));
    
    // Update state mean and covariance matrix
    x_ += K * z_diff;
    x_(3) = Tools::NormalizeAngle(x_(3));
    P_ -= K * S * K.transpose();

    // Calculate Normalized Innovation Squared (NIS).
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

// Calculates sigma points used in unscented transformation for approximation
// of real distribution of non-lenear motion model.
void UKF::CalcAugmentedSigmaPoints()
{
    // Predict measurement:
    
    //create augmented state covariance matrix
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    assert(P_.rows() == n_x_ && P_.cols() == n_x_);
    P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
    P_aug(n_std_a_, n_std_a_) = std_a_ * std_a_;
    P_aug(n_std_yawdd_, n_std_yawdd_) = std_yawdd_ * std_yawdd_;
    
    //create augmented mean state vector
    VectorXd x_aug = VectorXd::Zero(n_aug_); // mean values of std_a noise and std_yawdd noise are zero.
    x_aug.head(n_x_) = x_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();
    
    //create augmented sigma points
    // 0 | 1-7 | 8-15
    
    Xsig_aug_.col(0) = x_aug;
    
    L *= sqrt(lambda_ + n_aug_);
    
    for (int i = 0; i < n_aug_; i++)
    {
        Xsig_aug_.col(i + 1) = x_aug + L.col(i);
        Xsig_aug_.col(i + 1 + n_aug_) = x_aug - L.col(i);
    }
}

// Predicts sigma points for non-linear motion model using augmented sigma points.
void UKF::PredictSigmaPoints(double delta_t)
{
    // augmented state vector: [px, py, v, phi, phi-dot, nu-a, nu-phi-dot-dot]^t
    // model state vector: [px, py, v, phi, phi-dot]^t
    
    const auto delta_t2 = delta_t * delta_t;
    
    //write predicted sigma points into right column
    //predict sigma points
    for (int i = 0; i< n_sigma_; i++)
    {
        //extract values for better readability
		auto p_x = Xsig_aug_(0,i);
		auto p_y = Xsig_aug_(1,i);
		auto v = Xsig_aug_(2,i);
		auto yaw = Xsig_aug_(3,i);
		auto yawd = Xsig_aug_(4,i);
		auto nu_a = Xsig_aug_(5,i);
		auto nu_yawdd = Xsig_aug_(6,i);
        
        //predicted state values
        double px_p, py_p;
        
        //avoid division by zero
        if (fabs(yawd) > Tools::epsilon) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }
        
		auto v_p = v;
		auto yaw_p = yaw + yawd*delta_t;
		auto yawd_p = yawd;
        
        //add noise
        px_p += 0.5*nu_a*delta_t2 * cos(yaw);
        py_p += 0.5*nu_a*delta_t2 * sin(yaw);
        v_p = v_p + nu_a*delta_t;
        
        yaw_p += 0.5*nu_yawdd*delta_t2;
        yawd_p += nu_yawdd*delta_t;
        
        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }
}

// Predicts the state covariance matrix.
void UKF::PredictCovariance()
{
    //predict state covariance matrix
    P_.fill(0.);
    
    VectorXd x_diff(n_x_);
    for (int i = 0; i < n_sigma_; i++)
    {
        // state difference
        x_diff = Xsig_pred_.col(i) - x_;        
        x_diff(3) = Tools::NormalizeAngle(x_diff(3));
        
        P_ += weights_(i) * x_diff * x_diff.transpose();
    }
}

MatrixXd UKF::TransformRadarSigmaPointsIntoMeasurementSpace()
{
    MatrixXd Zsig(n_z_radar_, n_sigma_);
    
    //transform sigma points into measurement space
    for (int i = 0; i < n_sigma_; i++)
    {
        // extract values for better readibility
        auto px = Xsig_pred_(0, i);
		auto py = Xsig_pred_(1, i);
        const auto vel = Xsig_pred_(2, i);
        const auto yaw = Xsig_pred_(3, i);
        
        // Avoid division by zero
		int cnt = 0;
        if (fabs(px) <= Tools::epsilon){
            px = Tools::epsilon;
			cnt++;
        }
        if (fabs(py) <= Tools::epsilon){
            py = Tools::epsilon;
			cnt++;
        }
        
        const auto pho = sqrt(px * px + py * py);
        
        // measurement model
        Zsig(0, i) = pho;                                           //r
        Zsig(1, i) = atan2(py, px);                                 //phi
        Zsig(2, i) = (px * cos(yaw) *vel + py * sin(yaw) *vel) / pho;   //r_dot

		if (cnt == 2) {
			Zsig(1, i) = 0; // replace returned 45 degree by zero for the case when px=0 & py=0.
		}
    }
    
    return Zsig;
}

MatrixXd UKF::TransformLidarSigmaPointsIntoMeasurementSpace()
{
    MatrixXd Zsig(n_z_lidar_, n_sigma_);
    
    //transform sigma points into measurement space
    for (int i = 0; i < n_sigma_; i++)
    {
        // measurement model
        Zsig(0, i) = Xsig_pred_(0, i); // px
        Zsig(1, i) = Xsig_pred_(1, i); // py
    }
    
    return Zsig;
}

VectorXd UKF::CalcPredictedMean(const MatrixXd& Xsig_pred, const VectorXd& weights)
{
    VectorXd x_pred = VectorXd::Zero(Xsig_pred.rows());
    
    //calculate mean predicted measurement
    for (int i = 0; i < Xsig_pred.cols(); i++) {
        x_pred = x_pred + weights(i) * Xsig_pred.col(i);
    }
    
    return x_pred;
}
