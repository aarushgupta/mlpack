/**
 * @file huber_loss_impl.hpp
 * @author Aarush Gupta
 *
 * Implementation of the huber loss performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_HUBER_LOSS_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_HUBER_LOSS_IMPL_HPP

// In case it hasn't yet been included.
#include "huber_loss.hpp"

namespace mlpack{
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
HuberLoss<InputDataType, OutputDataType>::HuberLoss()
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
double HuberLoss<InputDataType, OutputDataType>::Forward(
    const InputType&& input, const TargetType&& target)
{
  return Fn(input, target);
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void HuberLoss<InputDataType, OutputDataType>::Backward(
    const InputType&& input, 
    const TargetType&& target,
    OutputType&& output)
{
  Deriv(input, target, output);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void HuberLoss<InputDataType, OutputDataType>::serialize(
    Archive& /* ar */,
    const unsigned int /* version */)
{
  // Nothing to do here.
}

} // namespace ann
} // namespace mlpack

#endif
