#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

//三角化两帧间某个对应特征点的深度
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

//PNP方法得到第l帧到第i帧的R_initial、P_initial
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)//要把待求帧i上所有特征点的归一化坐标和3D坐标(l系上)都找出来
	{
		if (sfm_f[j].state != true)//这个特征点没有被三角化为空间点，跳过这个点的PnP
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)//依次遍历特征j在每一帧中的归一化坐标
		{
			if (sfm_f[j].observation[k].first == i)//如果该特征在帧i上出现过
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);//得到了第i帧到第l帧的旋转平移
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);//转换成原有格式
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

//三角化frame0和frame1间所有对应点
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)//在所有特征里面依次寻找
	{
		if (sfm_f[j].state == true)//如果这个特征已经三角化过了，那就跳过
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)//如果这个特征在frame0出现过
			{
				point0 = sfm_f[j].observation[k].second;//把他的归一化坐标提取出来
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)//如果这个特征在frame1出现过
			{
				point1 = sfm_f[j].observation[k].second;//把他的归一化坐标提取出来
				has_1 = true;
			}
		}
		if (has_0 && has_1)//如果这两个归一化坐标都存在
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);//根据他们的位姿和归一化坐标，输出在参考系l下的的空间坐标
			sfm_f[j].state = true;//已经完成三角化，状态更改为true
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

/**
 * @brief   纯视觉sfm，求解窗口中的所有图像帧的位姿和特征点坐标
 * @param[in]   frame_num	窗口总帧数（frame_count + 1）
 * @param[out]  q 	窗口内图像帧的旋转四元数q（相对于第l帧）//q_w_ck
 * @param[out]	T 	窗口内图像帧的平移向量T（相对于第l帧）//T_w_ck
 * @param[in]  	l 	第l帧
 * @param[in]  	relative_R	当前帧到第l帧的旋转矩阵
 * @param[in]  	relative_T 	当前帧到第l帧的平移向量
 * @param[in]  	sfm_f		所有特征点
 * @param[out]  sfm_tracked_points 所有在sfm中三角化的特征点ID和坐标
 * @return  bool true:sfm求解成功
*/
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	//假设第l帧为原点，根据当前帧到第l帧的relative_R，relative_T，得到当前帧位姿
	//relative_R和relative_T设为T_l_ck
	//q[l]和T[l]设为T_w_l  即l为第一帧
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	//T_w_ck = T_w_l * T_l_ck  这里得到当前帧位姿T_w_ck
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);//frame_num-1表示当前帧
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];
	
	//（因为要三角化：这里以第l帧做参考系，就是说假设世界坐标系是l帧，而三角化输入的位姿是世界到相机系的，所以位姿是l帧到当前帧（世界->相机）

	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();//c_Rotation是第l帧到当前帧
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	//T_l_w
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];//这里的pose表示的是第l帧到每一帧的变换矩阵
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	//T_ck_w
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	//1、先三角化第l帧（参考帧）与第frame_num-1帧（当前帧）的路标点
	//2、pnp求解从第l+1开始的每一帧到第l帧的变换矩阵R_initial, P_initial，保存在Pose中
	//并与当前帧进行三角化
	for (int i = l; i < frame_num - 1 ; i++)
	{
		if (i > l)
		{
			//T_ck_w
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}

	//3、从第l+1帧到滑窗的最后的每一帧再与第l帧进行三角化补充3D坐标
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);


	//4、对于在sliding window里在第l帧之前的每一帧，分别都和后一帧用PnP求它的位姿，得到位姿后再和第l帧三角化得到它们共视点的3D坐标  
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}

	//5、三角化其他未恢复的特征点。
	//至此得到了滑动窗口中所有图像帧的位姿以及特征点的3d坐标
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/

	//6、使用cares进行全局BA优化
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//加入待优化量：全局位姿
		//在这里，可以发现，仅仅是位姿被优化了，特征点的3D坐标没有被优化！
		//T_ck_w
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		//固定先验值
		//因为l帧是参考系，最新帧的平移也是先验，如果不固定住，原本可观的量会变的不可观。
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 
		}

	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	
	//这里得到的是第l帧坐标系到各帧的变换矩阵，应将其转变为各帧在第l帧坐标系上的位姿
	//需要获得各帧在帧l系下的位姿(也就是各帧到l帧的旋转平移)，所以需要inverse操作，然后把特征点在帧l系下的3D坐标传递出来。
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		//T_l_ck
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		//T_l_ck
		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

