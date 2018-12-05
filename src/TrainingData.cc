//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panga contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include "TrainingData.h"
#include "RandomWrapper.h"

TrainingData& TrainingData::operator=(const TrainingData& rhs) {
    return *this;
}

void TrainingData::Shuffle(RandomWrapper* randomWrapper) {
    randomWrapper->ShuffleVector(this);
}
/*
void TrainingData::Scale() {
	if(0 == scale_training_data(this->scalingSettings, *this)) {
		this->scaled = true;
	}
}

void TrainingData::Descale() {
	// implementation TBD
    Assert(false, "Call to unimplemented TrainingData::Descale\n");

	this->scaled = false;
}

bool TrainingData::IsScaled() {
	return this->scaled;
}

void TrainingData::DescaleOutput(Vector<double>& out) {
	scale_data(out, this->scalingSettings, ScalingDataTypeOutput, ScalingModeDescale);
}

void TrainingData::ScaleInput(Vector<double>& in) {
	scale_data(in, this->scalingSettings, ScalingDataTypeInput, ScalingModeScale);
}

void TrainingData::DescaleInput(Vector<double>& in) {
	scale_data(in, this->scalingSettings, ScalingDataTypeInput, ScalingModeDescale);
}

int TrainingData::RemoveDataFraction(TrainingData& kept, TrainingData& removed, double fraction) {
	kept.clear();
	removed.clear();

	// kind of hax but we need to save these params in kept and removed 
	kept.scalingSettings = removed.scalingSettings = this->scalingSettings;
	kept.scaled = removed.scaled = this->scaled;

    for (unsigned int i = 0; i < this->num_examples; i++) {
        if (RandomCoinFlip(fraction))
        {
			removed.input.add(this->input[i]);
			removed.output.add(this->output[i]);
			removed.index.add(this->index[i]);
		} 
        else 
        {
			kept.input.add(this->input[i]);
			kept.output.add(this->output[i]);
			kept.index.add(this->index[i]);
		}
	}

	kept.num_input = removed.num_input = this->num_input;
	kept.num_output = removed.num_output = this->num_output;
	kept.num_examples = kept.input.length();
	removed.num_examples = removed.input.length();

	// if we failed to remove any data, try again
	if(removed.num_examples == 0) {
		return this->RemoveDataFraction(kept, removed, fraction);
	}

    Assert(kept.num_examples + removed.num_examples == this->num_examples, "We must have copied all the examples to either kept or removed.\n");

	return 0;
}
*/