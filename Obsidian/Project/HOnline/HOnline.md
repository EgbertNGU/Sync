## 环境配置
1. 修改cmake库路径配置
2. 修改**cudax/cuda_glm.h**->line:2 `#define CUDA_VERSION 7000``为#define CUDA_VERSION 114000`
>报错信息：
>home/ngu/Project/Gesture/honline/cudax/../cudax/functors/CorrespondencesFinder.h(30): error: calling a constexpr __host__ function("operator[]") from a __device__ function("print_vec3") is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.

3. 修改`#include "tracker/Hmodel/Model.h"`为`#include "tracker/HModel/Model.h"`
>**tracker/Energy/JointLimits.cpp**->line:2 
>**tracker/Detection/DetectionStream.h**->line:11
>tracker/Detection/QianDetection.cpp->line:10

4. 删除多余的 `#pragma once`
>**tracker/Energy/Fingertips.cpp**
>**tracker/Energy/ShapeSpace.cpp**
>**tracker/HModel/OutlineFinder.cpp**
>**tracker/OpenGL/ConvolutionRenderer/ConvolutionRenderer.cpp**
5. 删除**tracker/HModel/OutlineFinder.cpp**->line:5 `#include <windows.h>`
6. 将**apps/honline_experiments/experiments.cpp**->line:4 `#include <windows.h>`修改为`#include <unistd.h>`，其他问题未修改，未生成该可执行文件
7. 注释`#include <OpenGP/GL/EigenOpenGLSupport3.h>`
>**tracker/OpenGL/ConvolutionRenderer/ConvolutionRenderer.h**->line:7
>tracker/OpenGL/ConvolutionRenderer/ConvolutionRenderer.cpp->line:8
>报错信息：
>#include <OpenGP/GL/EigenOpenGLSupport3.h>

7. **tracker/HModel/Model.h**修改，删除`Model::`
>line:307 void Model::update(Phalange & phalange);
>line:309 const std::vector<float\> & Model::get_theta();
>line:311 const std::vector<float\>& Model::get_beta();
>line:317 std::vector<float\> Model::get_updated_parameters(const vector<float\> & theta, const vector<float\> &delta_theta);
8. 删除**tracker/Energy/ShapeSpace.h**->line:63 `Eigen::Matrix<double, Eigen::Dynamic, 1>compute_objective_double(const std::vector<double> & beta_double, int beta_index, const std::vector<double> & beta_latent_double, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & F);`中的`energy::ShapeSpace::`
9. 修改**tracker/GLWidget.h**，删除`GLWidget::`
>line:76 std::vector<std::pair<Vector3, Vector3>> GLWidget::prepare_data_correspondences_for_degub_renderer();
>line:78 std::vector<std::pair<Vector3, Vector3>> GLWidget::prepare_silhouette_correspondences_for_degub_renderer();
10. **tracker/HModel/BatchSolver.cpp**->增加`#include <unistd.h>`，将line:342 `Sleep(5000);`修改为`sleep(0.5);`
11. Linux系统下修改**tracker/HModel/SyntheticDatasetGenerator.h**
>增加include
>```C++
>#include <sys/stat.h>
>#include <sys/types.h>
>```
>line:78 `CreateDirectory(path.c_str(), NULL);` 修改为 `mkdir(path.c_str(), 0755);`
12. Linux系统下修改**tracker/Tracker.h**
>line:161 `Sleep(1500)`修改为 `sleep(1.5)`
>line:548  `CreateDirectory(path.c_str(), NULL);` 修改为 `mkdir(path.c_str(), 0755);`
>line:936 `Sleep(1000)`修改为 `sleep(1)`
>增加头文件：
>>`#include <unistd.h>`
>>`#include <sys/stat.h>`
>>`#include <sys/types.h>`

13. 修改**apps/honline_atb/main.cpp**
>line:91 `worker.settings->data_path`改为工程文件夹下的data文件夹
>line:113 `worker.settings->dataset_type = TKACH;`改为`worker.settings->dataset_type = INTEL;`

14.  修改相机参数`tracker/Data/Camera.cpp`
15. Linux系统下将`data/models/anastasia/B.txt`中32767的值全部改为2147483647。该值为RAND_MAX值，在windows系统下该值为32767，而在ubuntu系统下改值为2147483647。

## 问题及解决方法
1. 在线手模型注册阶段报错
	```
	terminate called after throwing an instance of 'thrust::system::system_error'
    what():  for_each: failed to synchronize: cudaErrorIllegalAddress: an illegal 
    memory access was encountered
	```
```
Eigen::DenseCoeffsBase<Derived, 0>::CoeffReturnType Eigen::DenseCoeffsBase<Derived, 0>::operator()(Eigen::Index, Eigen::Index) const [with Derived = Eigen::Matrix<float, -1, -1>; Eigen::DenseCoeffsBase<Derived, 0>::CoeffReturnType = const float&; Eigen::Index = long int]: 假设 ‘row >= 0 && row < rows() && col >= 0 && col < cols()’ 失败。
```


2. 某些时候无法正常启动
	与Eigen库的使用有关，报错信息: 
```
/usr/include/eigen3/Eigen/src/Core/DenseStorage.h:128：Eigen::internal::plain_array<T, Size, MatrixOrArrayOptions, 32>::plain_array() [with T = float; int Size = 16; int MatrixOrArrayOptions = 0]: 假设 ‘(internal::UIntPtr(eigen_unaligned_array_assert_workaround_gcc47(array)) & (31)) == 0 && "this assertion is explained here: " "http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html" " **** READ THIS WEB PAGE !!! ****"’ 失败。
```
Eigen 内存对齐，https://zhuanlan.zhihu.com/p/349413376
修改:
+ // tracker/HModel/Model.h->line:48 `std::vector<Vec3d> offsets;`改为`std::vector<Vec3d, Eigen::aligned_allocator<Vec3d>> offsets;`未生效
+ 更换为c++17解决

## 暂未解决但改动后有效
1. 

## 知识整理
1. C++`Sleep\sleep`函数
	`Sleep()`：在**Windows**下使用，头文件为`#include <windows.h>`，单位为毫秒(ms)
	`sleep()`：在Linux下使用，头文件为`#include <unistd.h>`，单位为秒(s)
2. Linux C++ 文件夹相关操作
	https://blog.csdn.net/Swallow_he/article/details/109639047?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-109639047-blog-122670188.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-109639047-blog-122670188.pc_relevant_default&utm_relevant_index=1
	

## 待研究
1. C++ 文件中判断系统类型
```C++
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)
//defined for 32 and 64-bit environments
#include <io.h>                // for _access(), _mktemp()
#define GP_MAX_TMP_FILES  27   // 27 temporary files it's Microsoft restriction
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
//all UNIX-like OSs (Linux, *BSD, MacOSX, Solaris, ...)
#include <unistd.h>            // for access(), mkstemp()
#define GP_MAX_TMP_FILES  64
#else
#error unsupported or unknown operating system
#endif
```

