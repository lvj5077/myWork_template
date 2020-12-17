#include <sys/types.h>
#include <sys/stat.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <cassert>
#include <cstring>
#include <eigen3/Eigen/Dense>


class Utility
{
  public:
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

  
};


#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <Eigen/Geometry>

#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

#include <Eigen/Dense>

#include "glog/logging.h"

using namespace Eigen;
using namespace std;
using namespace ceres;

extern const double data1[];
extern const double data2[]; 
const int kNumObservations = 9;

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}




struct ProjectionFactorNum
{
	ProjectionFactorNum(const Eigen::VectorXd &_pose_VIO, const Eigen::VectorXd &_pose_GT) : pose_VIO(_pose_VIO), pose_GT(_pose_GT)
	{
	}

	bool operator()(const double * const pare_T_cam, double * residuals) const
	{
        // Eigen::MatrixXd T44(4,4);
        // Eigen::MatrixXd TBT(4,4);
        // Eigen::MatrixXd A(4,4);
        // Eigen::MatrixXd B(4,4);
        Eigen::Matrix<double, 4, 4> T44 = Eigen::Matrix<double, 4, 4>::Identity(); 
        Eigen::Matrix<double, 4, 4> TBT = Eigen::Matrix<double, 4, 4>::Identity(); 
        Eigen::Matrix<double, 4, 4> A = Eigen::Matrix<double, 4, 4>::Identity();
        Eigen::Matrix<double, 4, 4> B = Eigen::Matrix<double, 4, 4>::Identity(); 


            // Eigen::MatrixXd T_resdual;
        Eigen::Matrix<double, 4, 4> T_residual = Eigen::Matrix<double, 4, 4>::Identity(); 


	    Eigen::Quaterniond q_A(pose_GT(6), pose_GT(3), pose_GT(4), pose_GT(5));

        A(0,3) = pose_GT(0); 
        A(1,3) = pose_GT(1); 
        A(2,3) = pose_GT(2);
        A.block(0,0,3,3) = q_A.toRotationMatrix();

	    Eigen::Quaterniond q_B(pose_VIO(6), pose_VIO(3), pose_VIO(4), pose_VIO(5));
        B(0,3) = pose_VIO(0); 
        B(1,3) = pose_VIO(1); 
        B(2,3) = pose_VIO(2);
        B.block(0,0,3,3) = q_B.toRotationMatrix();



	    Eigen::Quaterniond q_T_cam(pare_T_cam[6], pare_T_cam[3], pare_T_cam[4], pare_T_cam[5]);

        T44(0,3) = pare_T_cam[0]; 
        T44(1,3) = pare_T_cam[1]; 
        T44(2,3) = pare_T_cam[2];
        T44.block(0,0,3,3) = q_T_cam.toRotationMatrix();

	    TBT = (T44.inverse())*B*T44;


        T_residual = TBT.inverse()*A;

        // Eigen::Quaterniond q_tbt(TBT.block(0,0,3,3));
        Eigen::Vector3d t(T_residual(0,3),T_residual(1,3),T_residual(2,3));
        Sophus::SE3<double> SE3_Rt(T_residual.block(0,0,3,3), t); 
        typedef Eigen::Matrix<double,6,1> Vector6d;
        Vector6d se3 = SE3_Rt.log();

	    // residuals[0] = se3(0);
	    // residuals[1] = se3(1);
        // residuals[2] = se3(2);

	    residuals[0] = se3(3);
	    residuals[1] = se3(4);
        residuals[2] = se3(5);

	    residuals[3] = se3(3);
	    residuals[4] = se3(4);
        residuals[5] = se3(5);

        // // test 
	    // residuals[0] = pare_T_cam[0]-1.1;
	    // residuals[1] = pare_T_cam[1]-2.2;
        // residuals[2] = pare_T_cam[2]-3.3;
	    // residuals[3] = pare_T_cam[3]-4.4;
	    // residuals[4] = pare_T_cam[4]-5.5;
        // residuals[5] = pare_T_cam[5]-6.6;

        
    	return true;
	}

	static ceres::CostFunction* Create(const Eigen::VectorXd pose_VIO,
	                                   const Eigen::VectorXd pose_GT) 
	{
	  return (new ceres::NumericDiffCostFunction<
	          ProjectionFactorNum, ceres::CENTRAL, 6, 7>(
	          	new ProjectionFactorNum(pose_VIO,pose_GT)));
	}

	Eigen::VectorXd pose_VIO;
	Eigen::VectorXd pose_GT;

};



int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);


	// Tcam[0,3] = 0.04
	// Tcam[1,3] = 0.05
	// Tcam[2,3] = 0.06


    double T_cam[7];

    T_cam[0] = 0.022;
    T_cam[1] = -0.015;
    T_cam[2] = -0.014;


    T_cam[3] = 0.5;
    T_cam[4] = 0.5;
    T_cam[5] = 0.5;
    T_cam[6] = 0.5;

    ceres::Problem problem;


    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(T_cam, 7, local_parameterization);

    for (int i = 0; i < kNumObservations; ++i) {

        Eigen::VectorXd obs_vio(7);
        obs_vio << data1[8*i+1], data1[8*i+2],data1[8*i+3],data1[8*i+4],data1[8*i+5],data1[8*i+6],data1[8*i+7];
        Eigen::VectorXd obs_gt(7);
        obs_gt  << data2[8*i+1], data2[8*i+2],data2[8*i+3],data2[8*i+4],data2[8*i+5],data2[8*i+6],data2[8*i+7];

        cout << obs_vio.transpose() <<endl;  
        cout << obs_gt.transpose() <<endl; 
        cout << "~~~~~~~~~~~~~~~~~~~~"<<endl; 

        ceres::CostFunction *cost_function;
        cost_function =
                new ceres::NumericDiffCostFunction<ProjectionFactorNum, ceres::CENTRAL, 6,7>(
                        new ProjectionFactorNum(obs_vio, obs_gt));
        problem.AddResidualBlock(cost_function, NULL, T_cam);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 250;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    std::cout << T_cam[0] << " " << T_cam[1] << " " << T_cam[2] << " " << T_cam[3] << "," << T_cam[4] << "," << T_cam[5] << ","<< T_cam[6]  << "\n";
	// Tcam[0,3] = 0.04
	// Tcam[1,3] = 0.05
	// Tcam[2,3] = 0.06
    return 0;
}


const double data2[] = {
// timestamp, x,y,z,qx,qy,qz,qw
143.956000999999,-0.249068586447442,-0.102648613608117,-0.131260657408835,-0.104943175376519,0.194098634683653,0.0452345236088096,0.974303078015111,
128.622989,-0.264368738398623,0.00537874041028999,-0.0936893055471126,-0.020673472551782,0.218573777491412,0.0233588422320125,0.97532172938747,
115.623262000001,-0.287349637628682,0.132228539969345,-0.0660767933731697,0.0802735239336835,0.227161160651098,-0.00140246327363925,0.97054211734663,
81.8829239999977,-0.0538893447802521,0.0973853055492008,0.0148239232948193,0.0886807291872089,0.0589202735312856,-0.0186072256656494,0.994141791089607,
70.0424199999979,-0.0465302246635883,-0.137923985633823,-0.0756215550618891,-0.133684543617314,0.0496255696405536,-0.004712368062503,0.989769437405967,
58.7022179999985,-0.0477212605605165,-0.0841638678421194,-0.0412146059648899,-0.0850006290156296,0.0444372294941342,-0.00849463410940299,0.995353237244537,
35.021209999999,0.0864284143866529,0.096872321105171,0.0119955957030675,0.0585233811982345,-0.0257086621596341,-0.0652967430522906,0.995816455924409,
23.0139380000001,0.0909954163423154,-0.00533822494309341,-0.0106435241024031,-0.00685968494379164,-0.0362406195148452,-0.0688363192402832,0.996945897916784,
10.8398979999984,-0.170627812765173,0.0357214854119547,-0.0442205397847337,0.00920602128542006,0.159478002209545,0.0730296802715183,0.984453493966467
};


const double data1[] = {
144.491667,0.030325,0.002051,0.046389,-0.106305,0.173327,0.044492,0.978099,
130.933333,0.020547,-0.002959,0.020832,-0.007642,0.191886,0.02451,0.981081,
118.525,0.008016,-0.002092,-0.007896,0.099976,0.210579,0.00183,0.972449,
83.316667,-0.003421,0.001785,-0.018093,0.074815,0.038089,-0.018533,0.996297,
68.791667,0.001775,0.006231,0.036438,-0.107669,0.035946,-0.007863,0.993506,
60.066667,0.000684,0.003548,0.023136,-0.062904,0.036833,-0.010164,0.997288,
35.075,-0.017367,0.009702,-0.029995,0.077299,-0.056518,-0.069162,0.992999,
23.258333,-0.019134,0.009408,-0.007461,0.001747,-0.053133,-0.071983,0.995988,
10.191667,0.024716,-0.006941,0.009297,0.00743,0.113188,0.071833,0.990946

};