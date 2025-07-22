#pragma once

#include "ceres/ceres.h"
#include <ceres/rotation.h>
#include <glm/glm.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <set>

using Eigen::Vector3d;

namespace CeresSolver {

struct EqualityConstraint
{
	EqualityConstraint(const Vector3d& point_, double weight_)
		: point(point_), weight(weight_)
	{
	}

	template<typename T>
	bool operator()(const T* const pos, T* residual) const
	{
		// TODO: Implement the cost function
		residual[0] = T(weight) * (T(point(0)) - pos[0]);
        residual[1] = T(weight) * (T(point(1)) - pos[1]);
        residual[2] = T(weight) * (T(point(2)) - pos[2]);

		return true;
	}

    static ceres::CostFunction* create(const Vector3d& targetPoint, const double weight) {
    return new ceres::AutoDiffCostFunction<EqualityConstraint, 3, 3>(
        new EqualityConstraint(targetPoint, weight)
        );
    }

private:
	const Vector3d point;
    double weight;
};


struct EnergyCostFunction
{
	EnergyCostFunction(const Vector3d& p_i_, const Vector3d& p_j_, double weight_)
		: delta_ij(p_i_ - p_j_), weight(weight_)
	{
	}

	template<typename T>
	bool operator()(const T* const p_i_prime, const T* const p_j_prime, const T* const angle, T* residual) const
	{
        T inputPoint[3];
        inputPoint[0] = p_i_prime[0] - p_j_prime[0];
        inputPoint[1] = p_i_prime[1] - p_j_prime[1];
        inputPoint[2] = p_i_prime[2] - p_j_prime[2];

        T pos[3];
		T delta[3];
		delta[0] = T(delta_ij(0));
		delta[1] = T(delta_ij(1));
		delta[2] = T(delta_ij(2));

        ceres::AngleAxisRotatePoint(angle, inputPoint, pos);

		// TODO: Implement the cost function
        residual[0] = T(weight) * (delta[0] - pos[0]);
        residual[1] = T(weight) * (delta[1] - pos[1]);
        residual[2] = T(weight) * (delta[2] - pos[2]);

		return true;
	}

    static ceres::CostFunction* create(const Vector3d& p_i, const Vector3d& p_j, const double weight) {
    return new ceres::AutoDiffCostFunction<EnergyCostFunction, 3, 3, 3, 3>(
        new EnergyCostFunction(p_i, p_j, weight)
        );
    }

private:
	const Vector3d delta_ij;
    double weight;
};




}    