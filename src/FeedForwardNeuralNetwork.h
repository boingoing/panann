//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for
// full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef FEEDFORWARDNEURALNETWORK_H__
#define FEEDFORWARDNEURALNETWORK_H__

#include <vector>

#include "Perceptron.h"

namespace panann {

class TrainingData;

/**
 * A feed-forward neural network supporting training via multiple algorithms.
 */
class FeedForwardNeuralNetwork : public Perceptron {
 public:
  enum class TrainingAlgorithmType : uint8_t {
    /**
     * Backpropagation with momentum and learning rate parameters.<br/>
     * This is an online learning algorithm and does not perform batching weight
     * updates.
     * @see SetLearningRate
     * @see SetMomentum
     */
    Backpropagation = 0,
    /**
     * Batching backpropagation with learning rate parameter.<br/>
     * This is an offline learning algorithm. It batches together weight updates
     * and modifies the weights at the end of each epoch.
     * @see SetLearningRate
     */
    BatchingBackpropagation,
    /**
     * An implementation of quickprop.<br/>
     * It uses the learning rate, mu, and qprop weight decay parameters.<br/>
     * This is an offline learning algorithm.
     * @see SetLearningRate
     * @see SetQpropMu
     * @see SetQpropWeightDecay
     */
    QuickBackpropagation,
    /**
     * Resilient backprop is a very fast training algorithm designed to move
     * quickly down the error curve when the derivative of the partial error
     * doesn't change.<br/> The implementation here is of iRPROP-. It is an
     * offline learning algorithm.
     * @see SetRpropWeightStepInitial
     * @see SetRpropWeightStepMin
     * @see SetRpropWeightStepMax
     * @see SetRpropIncreaseFactor
     * @see SetRpropDecreaseFactor
     */
    ResilientBackpropagation,
    /**
     * Simulated annealing and weight decay are added to resilient
     * backprop.<br/> Attempts to avoid getting stuck in local minima on the
     * error surface while training the network.<br/> Uses all of the rprop
     * parameters plus some specific to sarprop.
     * @see SetRpropWeightStepInitial
     * @see SetRpropWeightStepMin
     * @see SetRpropWeightStepMax
     * @see SetRpropIncreaseFactor
     * @see SetRpropDecreaseFactor
     * @see SetSarpropWeightDecayShift
     * @see SetSarpropStepThresholdFactor
     * @see SetSarpropStepShift
     * @see SetSarpropTemperature
     */
    SimulatedAnnealingResilientBackpropagation
  };

  FeedForwardNeuralNetwork() = default;
  FeedForwardNeuralNetwork(const FeedForwardNeuralNetwork&) = delete;
  FeedForwardNeuralNetwork& operator=(const FeedForwardNeuralNetwork&) = delete;
  ~FeedForwardNeuralNetwork() override = default;

  /**
   * Set the learning rate parameter used by backprop, batch, and qprop.<br/>
   * Default: 0.7
   */
  void SetLearningRate(double learning_rate);
  double GetLearningRate() const;

  /**
   * Set the momentum parameter used by backprop.<br/>
   * Default: 0.1
   */
  void SetMomentum(double momentum);
  double GetMomentum() const;

  /**
   * Set the Mu parameter for qprop.<br/>
   * Default: 1.75
   */
  void SetQpropMu(double mu);
  double GetQpropMu() const;

  /**
   * Set the weight decay parameter for qprop.<br/>
   * Default: -0.0001
   */
  void SetQpropWeightDecay(double weight_decay);
  double GetQpropWeightDecay() const;

  /**
   * Set the initial weight step value for rprop and sarprop.<br/>
   * This is sometimes called delta_zero.<br/>
   * Default: 0.0125
   */
  void SetRpropWeightStepInitial(double weight_step);
  double GetRpropWeightStepInitial() const;

  /**
   * Set the minimum weight step value for rprop and sarprop.<br/>
   * This is sometimes called delta_min.<br/>
   * If the weight change becomes 0, learning will stop so we need
   * to use some small value instead of zero.<br/>
   * Default: 0.000001
   */
  void SetRpropWeightStepMin(double weight_step);
  double GetRpropWeightStepMin() const;

  /**
   * Set the maximum weight step value for rprop and sarprop.<br/>
   * This is sometimes called delta_max.<br/>
   * If the weight change becomes too large, learning will be chaotic
   * so we clamp it to some reasonably large number to ensure smooth
   * training.<br/>
   * Default: 50
   */
  void SetRpropWeightStepMax(double weight_step);
  double GetRpropWeightStepMax() const;

  /**
   * Set the factor by which the weight step will be increased each
   * training step when the sign of the partial derivative of the error
   * does not change.<br/>
   * This is sometimes referred to as eta+.<br/>
   * Both rprop and sarprop use this paramter for the same purpose.<br/>
   * A higher value can increase the speed of convergeance but can also
   * make training unstable.<br/>
   * Default: 1.2
   */
  void SetRpropIncreaseFactor(double factor);
  double GetRpropIncreaseFactor() const;

  /**
   * Set the factor by which the weight step will be decreased when the
   * sign of the partial derivative of the error changes.<br/>
   * This is sometimes called eta-.<br/>
   * Both rprop and sarprop use this paramter for the same purpose.<br/>
   * A lower value will make training slower and a higher value can
   * make training unstable.<br/>
   * Default: 0.5
   */
  void SetRpropDecreaseFactor(double factor);
  double GetRpropDecreaseFactor() const;

  /**
   * Set the weight decay shift parameter used by sarprop.<br/>
   * This is a constant used as a weight decay.<br/>
   * It is called k1 in some formulations of sarprop.<br/>
   * Default: 0.01
   */
  void SetSarpropWeightDecayShift(double k1);
  double GetSarpropWeightDecayShift() const;

  /**
   * Set the weight step threshold factor parameter used by sarprop.<br/>
   * It is called k2 in some formulations of sarprop.<br/>
   * Default: 0.1
   */
  void SetSarpropStepThresholdFactor(double k2);
  double GetSarpropStepThresholdFactor() const;

  /**
   * Set the step shift parameter used by sarprop.<br/>
   * It is called k3 in some formulations of sarprop.<br/>
   * Default: 3
   */
  void SetSarpropStepShift(double k3);
  double GetSarpropStepShift() const;

  /**
   * Set the temperature parameter used by sarprop.<br/>
   * This is referred to as T in most sarprop formulations.<br/>
   * Default: 0.015
   */
  void SetSarpropTemperature(double t);
  double GetSarpropTemperature() const;

  /**
   * Set the training algorithm this network will use during training.<br/>
   * Default: ResilientBackpropagation
   * @see TrainingAlgorithmType
   */
  void SetTrainingAlgorithmType(TrainingAlgorithmType type);
  TrainingAlgorithmType GetTrainingAlgorithmType() const;

  /**
   * Use the training algorithm to train the network.<br/>
   * Training follows these steps:<br/>
   *   - For each example in the training data<br/>
   *   - Run the network forward on the example input<br/>
   *   - Calculate the total error by comparing output neuron values against the
   * example output<br/>
   *   - Calculate the partial error contributed by each weight in the
   * network<br/>
   *   - Update all the weights in the network to reduce the total error<br/>
   * Execute the above once for each epoch.<br/>
   * The actual method by which we will update the weights depends on the
   * training algorithm chosen.
   * @param training_data Examples on which we will train the network.<br/>
   * Note: Each epoch, shuffles the order of examples in training_data before
   * performing the training operation.
   * @param epoch_count The number of epochs we should execute to train the
   * network. One epoch is one full step through all of the training examples.
   * @see SetTrainingAlgorithmType
   * @see TrainingAlgorithmType
   * @see TrainingData
   */
  void Train(TrainingData* training_data, size_t epoch_count);

 protected:
  void UpdateSlopes();
  void UpdateWeightsOnline();
  void UpdateWeightsOffline(size_t current_epoch, size_t step_count);
  void UpdateWeightsBatchingBackpropagation(size_t step_count);
  void UpdateWeightsQuickBackpropagation(size_t step_count);
  void UpdateWeightsResilientBackpropagation();
  void UpdateWeightsSimulatedAnnealingResilientBackpropagation(
      size_t current_epoch);

  void ResetWeightSteps();
  void ResetSlopes();
  void ResetPreviousSlopes();

  void TrainOffline(TrainingData* training_data, size_t epoch_count);
  void TrainOnline(TrainingData* training_data, size_t epoch_count);

 private:
  static constexpr double DefaultLearningRate = 0.7;
  static constexpr double DefaultMomentum = 0.1;
  static constexpr double DefaultQpropMu = 1.75;
  static constexpr double DefaultQpropWeightDecay = -0.0001;
  static constexpr double DefaultRpropWeightStepInitial = 0.0125;
  static constexpr double DefaultRpropWeightStepMin = 0.000001;
  static constexpr double DefaultRpropWeightStepMax = 50;
  static constexpr double DefaultRpropIncreaseFactor = 1.2;
  static constexpr double DefaultRpropDecreaseFactor = 0.5;
  static constexpr double DefaultSarpropWeightDecayShift = 0.01;
  static constexpr double DefaultSarpropStepThresholdFactor = 0.1;
  static constexpr double DefaultSarpropStepShift = 3;
  static constexpr double DefaultSarpropTemperature = 0.015;

  std::vector<double> previous_weight_steps_;
  std::vector<double> slopes_;
  std::vector<double> previous_slopes_;

  double learning_rate_ = DefaultLearningRate;
  double momentum_ = DefaultMomentum;
  double qprop_mu_ = DefaultQpropMu;
  double qprop_weight_decay_ = DefaultQpropWeightDecay;
  double rprop_weight_step_initial_ = DefaultRpropWeightStepInitial;
  double rprop_weight_step_min_ = DefaultRpropWeightStepMin;
  double rprop_weight_step_max_ = DefaultRpropWeightStepMax;
  double rprop_increase_factor_ = DefaultRpropIncreaseFactor;
  double rprop_decrease_factor_ = DefaultRpropDecreaseFactor;
  double sarprop_weight_decay_shift_ = DefaultSarpropWeightDecayShift;
  double sarprop_step_threshold_factor_ = DefaultSarpropStepThresholdFactor;
  double sarprop_step_shift_ = DefaultSarpropStepShift;
  double sarprop_temperature_ = DefaultSarpropTemperature;

  TrainingAlgorithmType training_algorithm_type_ =
      TrainingAlgorithmType::ResilientBackpropagation;
};

}  // namespace panann

#endif  // FEEDFORWARDNEURALNETWORK_H__
