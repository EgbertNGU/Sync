离线数据运行报错
segment default：
GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(T _x, T _y)

运行存在异常的位置
1. void OutlineFinder::compute_projections_outline(const std::vector<glm::vec3> & centers, const std::vector<float> & radii, const std::vector<glm::vec3> & points, const glm::vec3 & camera_ray)
2. void HandFinder::binary_classification(cv::Mat& depth, cv::Mat& color, bool is_hand_exist, Eigen::MatrixXf joints_det)
3. void energy::Fitting::track(DataFrame& frame, LinearSystem& system, bool rigid_only, bool eval_error, bool calibrate, float & push_error, float & pull_error, float & weighted_error, int iter)
4. Matrix_MxN FindFingers::crop_image(const Matrix_MxN &depth, size_t & min_row, size_t & max_row, size_t & min_column, size_t & max_column)
	