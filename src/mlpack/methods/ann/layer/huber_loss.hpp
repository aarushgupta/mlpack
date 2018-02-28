/**
 * @file huber_loss.hpp
 * @author Aarush Gupta
 *
 * Definition of huber loss performance function
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HUBER_LOSS_HPP
#define MLPACK_METHODS_ANN_LAYER_HUBER_LOSS_HPP

#include <mlpack/prereqs.hpp>
#include <math.h>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The huber loss performance function measures the network's
 * performance according to a squared error or linear error
 * based on a threshold value provided.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class HuberLoss
{
 public:
  /**
   * Create the HuberLoss object.
   */
  HuberLoss(const double alpha = 1);

  /*
   * Computes the huber loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The actual labels/ desired output.
   */
  template<typename InputType, typename TargetType>
  double Forward(const InputType&& input, const TargetType&& target);
  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType&& input,
                const TargetType&& target,
                OutputType&& output);

  //! Get the input parameter.
  InputDataType& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the threshold.
  double const& Alpha() const { return alpha; }
  //! Modify the threshold.
  double& Alpha() { return alpha; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  /**
   *
   * Computes the huber loss  
   *
   * @param x input
   * @param y target 
   * @return the huber loss
   */
  double Fn(const double x, const double y)
  {
    double error = fabs(x - y);
    if error > alpha
      return alpha * ( error = 0.5 * alpha);
    else 
      return 0.5*(pow(error, 2));
  }

  /**
   *
   * Computes the huber loss  
   *
   * @param x input
   * @param y target 
   * @return the huber loss
   */
  template<typename eT>
  double Fn(const arma::Mat<eT>& x, arma::Mat<eT>& y)
  {
    double sum_error = 0;
    for (size_t i = 0; i < x.n_elem; i++)
    {
      sum_error = sum_error + Fn(x(i), y(i));
    }

    return sum_error;
  }

  /**
   *Computer the first derivative of the huber loss function
   *
   * @param x Input
   * @param y Target
   * @return  The resulting derivative
   */
  double Deriv(const double x, const double y)
  {
    double error = fabs(x - y);
    if (error < aplha || error = alpha)
      return (x - y);
    else
      return (x-y)/error;
  }

  /**
   *Computes the first derivative of the huber loss function
   *
   * @param x Input
   * @param y Target
   * @param o The resulting derivatives
   */
  template<typename InputType, typename TargetType, typename OutputType>  
  void Deriv(const InputType& x, const TargetType& y, OutputType& o)
  {
    for (size_t i = 0; i < x.n_elem; i++)
    {
      o(i) = Deriv(x(i), y(i));
    }
  }

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Threshold parameter
  double alpha;
}; // class HuberLoss

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "huber_loss_impl.hpp"

#endif
