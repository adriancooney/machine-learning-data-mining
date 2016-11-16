package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
)

func main() {
	// Parse in the illness.csv data. The second boolean argument denotes
	// whether the data has a CSV header or not.
	rawData, err := base.ParseCSVToInstances("../data/illness-mapped.csv", true)

	if err != nil {
		// Exit the program if we have an error importing the data
		panic(err)
	}

	title("Raw Data")
	fmt.Println(rawData)

	svm := ensemble.NewMultiLinearSVC("l1", "l2", true, 1.0, 1e-4, nil)

	// holdout(rawData, svm)
	crossFold(rawData, svm)
}

func holdout(rawData base.FixedDataGrid, svm *ensemble.MultiLinearSVC) {
	// Split the data up into training data and test data
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.5)

	svm.Fit(trainData)

	// Calculates the Euclidean distance and returns the most popular label
	predictions, _ := svm.Predict(testData)

	title("Predictions")
	fmt.Println(predictions)

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)

	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}

	title("Summary")
	fmt.Println(evaluation.GetSummary(confusionMat))
}

type conf struct {
	tp, fp, tn, prec, rec, f1 float64
	l                         int
}

func crossFold(rawData base.FixedDataGrid, svm base.Classifier) {
	confusionMatrices, _ := evaluation.GenerateCrossFoldValidationConfusionMatrices(rawData, svm, 10)

	// Save the averages
	data := make(map[string]*conf)

	var acc float64

	fmt.Println("Reference Class\tTrue Positives\tFalse Positives\tTrue Negatives\tPrecision\tRecall\tF1 Score")
	fmt.Println("---------------\t--------------\t---------------\t--------------\t---------\t------\t--------")

	for _, c := range confusionMatrices {
		for k := range c {
			d, ok := data[k]

			if !ok {
				d = &conf{}
				data[k] = d
			}

			d.tp += evaluation.GetTruePositives(k, c)
			d.fp += evaluation.GetFalsePositives(k, c)
			d.tn += evaluation.GetTrueNegatives(k, c)
			d.prec += evaluation.GetPrecision(k, c)
			d.rec += evaluation.GetRecall(k, c)
			d.f1 += evaluation.GetF1Score(k, c)
			d.l++
		}

		acc += evaluation.GetAccuracy(c)
	}

	for k, d := range data {
		l := float64(d.l)
		fmt.Printf("%s\t%.0f\t%.0f\t%.0f\t%.4f\t%.4f\t%.4f\n",
			k, d.tp/l, d.fp/l, d.tn/l, d.prec/l, d.rec/l, d.f1/l)
	}

	fmt.Printf("Overall accuracy: %.4f\n", acc/float64(len(confusionMatrices)))

	mean, variance := evaluation.GetCrossValidatedMetric(confusionMatrices, evaluation.GetAccuracy)
	fmt.Printf("Mean: %v\nVariance: %v", mean, variance)
}

func title(t string) {
	fmt.Printf("\n------ %s ------\n", t)
}
