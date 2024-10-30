#' Compute Cell Type Proportions
#' 
#' Computes the proportion of each cell type within each sample. Optionally, the
#' proportions can be logit-transformed to avoid extreme values.
#' @param cells Data frame or SingleCellExperiment/SpatialExperiment/SegmentedCells object.
#' @param feature Character; name of the column with cell type labels, default is "clusters".
#' @param imageID Character; name of the column with sample IDs, default is "sample_id".
#' @param logit Logical; if TRUE, applies logit transformation to proportions, default is TRUE.
#' @return Data frame of cell type proportions, with samples as rows and cell types as columns.
#' @importFrom SummarizedExperiment colData
#' @importFrom boot logit
#' @noRd
getProp <- function(cells, feature = "clusters", imageID = "sample_id", logit = TRUE) {
  
  # Extract relevant data based on the input class type
  if (is.data.frame(cells)) {
    df <- cells[, c(imageID, feature)]
  } else if (inherits(cells, "SingleCellExperiment") | inherits(cells, "SpatialExperiment")) {
    df <- as.data.frame(SummarizedExperiment::colData(cells))[, c(imageID, feature)]
  } else if (inherits(cells, "SegmentedCells")) {
    cellSummary <- cellSummary(cells, bind = TRUE)
    df <- as.data.frame(cellSummary[, c(imageID, feature)])
  } else {
    stop("Unsupported data type for 'cells'")
  }
  
  # Compute cell type proportions
  tab <- table(df[, imageID], df[, feature])
  tab <- sweep(tab, 1, rowSums(tab), "/")
  
  # Apply logit transformation if required
  if (logit) {
    try(tab[tab == 0] <- 0.001, silent = TRUE)
    try(tab[tab == 1] <- 0.999, silent = TRUE)
    tab <- boot::logit(tab)
  }
  
  as.data.frame.matrix(tab)
}

#' Compute Cell Type Proportions
#' 
#' Computes proportions for a specified cell type feature in a data frame,
#' SingleCellExperiment, or SpatialExperiment object. Allows for logit transformation.
#' @param cells A data frame, SingleCellExperiment, or SpatialExperiment object containing cell data.
#' @param feature Character; name of the feature column representing cell types.
#' @param imageID Character; name of the sample identifier column.
#' @param logit Logical; if TRUE, applies logit transformation to proportions.
#' @return A data frame with cell type proportions for each sample.
#' @importFrom boot logit
#' @noRd
computeProportion <- function(cells, feature = "clusters", imageID = "sample_id", logit = TRUE) {
  
  # Extract relevant columns based on input object type
  if (is.data.frame(cells)) {
    df <- cells[, c(imageID, feature)]
  } else if (is(cells, "SingleCellExperiment") || is(cells, "SpatialExperiment")) {
    df <- as.data.frame(SummarizedExperiment::colData(cells))[, c(imageID, feature)]
  } else if (is(cells, "SegmentedCells")) {
    cellSummary <- cellSummary(cells, bind = TRUE)
    df <- as.data.frame(cellSummary[, c(imageID, feature)])
  } else {
    stop("Unsupported cell object type")
  }
  
  # Calculate cell type proportions
  tab <- table(df[, imageID], df[, feature])
  tab <- sweep(tab, 1, rowSums(tab), "/")
  
  # Apply logit transformation if specified
  if (logit) {
    tab[tab == 0] <- 0.001  # Avoid infinite values for zero proportions
    tab[tab == 1] <- 0.999  # Avoid infinite values for one proportions
    tab <- boot::logit(tab)
  }
  
  return(as.data.frame.matrix(tab))
}


#' Compute Reference Sample Based on Covariance Norms
#' 
#' This function computes covariance matrices for each sample, then calculates 
#' pairwise Frobenius norms between them to identify reference samples.
#' Returns the top, bottom, median, reference, and validation samples.
#' @param data Data frame containing sample information.
#' @param markers Character vector of marker column names.
#' @param sampleCol Character; name of the sample ID column.
#' @param N Integer; number of top and bottom samples to return.
#' @return A list containing the matrix of norms, average norms, and sample indices
#' for top, bottom, median, reference, and validation samples.
#' @importFrom Rfast cova Norm
#' @noRd
computeReferenceSample <- function(data, markers, sampleCol = "sample_id", N = 2) {
  # Initialize containers for covariance matrices and norm calculations
  covMats <- list()
  samples <- unique(data[[sampleCol]])
  numSamples <- length(samples)
  norms <- matrix(0, nrow = numSamples, ncol = numSamples)
  
  # Calculate covariance matrices for each sample
  message("Calculating covariance matrices for each sample")
  for (i in seq_len(numSamples)) {
    sampleData <- data[data[[sampleCol]] == samples[[i]], markers]
    covMats[[i]] <- Rfast::cova(as.matrix(sampleData))
  }
  
  # Compute pairwise Frobenius norms
  message("Computing pairwise Frobenius norms between samples")
  for (i in seq_len(numSamples)) {
    for (j in seq_len(numSamples)) {
      covDiff <- covMats[[i]] - covMats[[j]]
      norms[i, j] <- Rfast::Norm(covDiff, type = "F")
      norms[j, i] <- norms[i, j]  # Ensure symmetry
    }
  }
  
  # Compute average norms and identify key samples
  avgNorms <- colMeans(norms, na.rm = TRUE)
  sortedIndices <- order(avgNorms)
  
  list(
    Norms = norms,
    avgNorms = avgNorms,
    topNSamples = samples[sortedIndices[seq_len(N)]],
    bottomNSamples = samples[sortedIndices[(numSamples - N + 1):numSamples]],
    medianSampleInd = samples[which(avgNorms == median(avgNorms))],
    refSampleInd = samples[which.min(avgNorms)],
    valSampleInd = samples[which.max(avgNorms)]
  )
}

#' Generate Heatmap Matrix for Marker Means by Cell Type
#' 
#' Creates a standardized heatmap matrix displaying the mean marker expression 
#' per cell type cluster. The heatmap matrix is clipped at a specified threshold 
#' and standardized to highlight expression differences.
#' @param data Data frame or matrix with marker expression values.
#' @param markers Character vector of marker columns to include. If NULL, all numeric columns are used.
#' @param clusters Vector of cluster labels for each row in the data.
#' @param threshold Numeric; the maximum absolute value for standard deviation clipping.
#' @param clusterMarkers Logical; if TRUE, clusters markers in the heatmap.
#' @param fontSize Numeric; font size for heatmap labels.
#' @return A matrix ready for heatmap visualization.
#' @importFrom stats aggregate
#' @noRd
generateHeatmapMatrix <- function(data, markers = NULL, clusters = NULL, threshold = 2,
                                  clusterMarkers = FALSE, fontSize = 14) {
  
  # Validate clusters input
  if (is.null(clusters)) {
    stop("Please provide a vector of cluster labels.")
  }
  
  # Validate data input
  if (!is.matrix(data) && !is.data.frame(data)) {
    stop("Input data must be a matrix or data frame.")
  }
  
  # Select all numeric columns if markers are not specified
  if (is.null(markers)) {
    if (!all(sapply(data, is.numeric))) {
      stop("If no markers are provided, all columns in data must be numeric.")
    }
    message("No markers provided; using all numeric columns as markers.")
    markers <- colnames(data)
  }
  
  # Subset the data to include only the selected markers
  markerData <- data[, markers, drop = FALSE]
  
  # Aggregate mean marker expression by cluster
  featuresHeatmap <- aggregate(. ~ clusters, data = cbind(clusters = clusters, markerData), FUN = mean)
  rownames(featuresHeatmap) <- featuresHeatmap[, 1]
  featuresHeatmap <- featuresHeatmap[, -1]  # Remove the clusters column after setting rownames
  
  # Standardize each column to center at 0 and scale by standard deviation
  featuresHeatmap <- sweep(featuresHeatmap, 2, colMeans(featuresHeatmap), "-")
  featuresHeatmap <- sweep(featuresHeatmap, 2, apply(featuresHeatmap, 2, sd), "/")
  
  # Apply threshold for clipping high/low values
  featuresHeatmap[featuresHeatmap > threshold] <- threshold
  featuresHeatmap[featuresHeatmap < -threshold] <- -threshold
  
  # Create a data frame for row annotations
  annotationRow <- data.frame(Clusters = rownames(featuresHeatmap))
  rownames(annotationRow) <- rownames(featuresHeatmap)
  
  # Identify row gaps for clustering groups in the heatmap
  gapRows <- which(!duplicated(substr(rownames(featuresHeatmap), 1, 2)))[-1] - 1
  
  # Uncomment to use pheatmap for visualization
  # pHeat <- ggplotify::as.ggplot(pheatmap(featuresHeatmap,
  #                                        gaps_row = gapRows,
  #                                        annotation_row = annotationRow, annotation_legend = FALSE,
  #                                        cluster_cols = clusterMarkers, cluster_rows = FALSE,
  #                                        fontsize = fontSize
  # ))
  # return(pHeat)
  
  return(as.matrix(featuresHeatmap))
}

#' Compute Elbow Point for BIC Plot
#' 
#' Identifies the optimal elbow point on a BIC plot by calculating the largest
#' difference between consecutive values. This is often used for selecting
#' the best alpha in group lasso.
#' @param vals Numeric vector of BIC values.
#' @return Integer; index of the optimal elbow point.
#' @noRd
computeElbow <- function(vals) {
  # Calculate differences between consecutive BIC values
  differences <- diff(vals)
  
  # Identify the index with the maximum absolute difference
  optimalIndex <- which.max(abs(differences)) + 1
  
  return(optimalIndex)
}

#' Train Cell Type Classifier using Caret
#' 
#' Trains a cell type classification model using LDA (or specified model) and
#' evaluates it on test data. Uses cross-validation for training.
#' @param trainX Data frame of training features with cell type labels.
#' @param testX Data frame of test features.
#' @param model Character; the method to use for training, e.g., "lda".
#' @return Predicted cell types for the test set.
#' @importFrom caret train trainControl
#' @noRd
trainCellTypeClassifier <- function(trainX, testX, model = "lda") {
  library(doParallel)
  library(caret)
  
  # Set up cross-validation control
  fitControl <- trainControl(method = "cv", number = 3)
  
  # Fit the model with specified method
  message("Fitting cell type classification model")
  set.seed(1994)
  classifierFit <- caret::train(cellTypes ~ ., data = trainX, method = model,
                                trControl = fitControl, trace = TRUE, preprocess = c("range"))
  print(classifierFit)
  
  # Predict cell types on the test data
  message("Predicting cell types on test data")
  predictedCellTypes <- predict(classifierFit, testX)
  
  return(predictedCellTypes)
}

#' Generate Hierarchical Clustering Tree for Group Lasso
#' 
#' Constructs a hierarchical clustering tree from feature correlations for use
#' in group lasso. Correlations are converted to distances, and the tree is
#' created using the specified clustering method.
#' @param features Numeric matrix of feature data.
#' @param method Character; clustering method, default is "ward".
#' @return List containing the hierarchical tree structure and node order.
#' @importFrom coop pcor
#' @importFrom ggtree ggtree
#' @importFrom stats as.dist hclust
#' @noRd
generateTree <- function(features, method = "ward") {
  # Compute the pairwise correlation distance matrix
  distanceMatrix <- coop::pcor(features) %>%
    cor2dist() %>%
    as.dist()
  
  # Generate the hierarchical clustering tree
  hcTree <- hclust(distanceMatrix, method = method)
  
  # Extract the clustering order
  hclustObj <- hcTree
  
  # Find child nodes within the tree for dendrogram representation
  hcTree <- treekoR:::findChildren(
    ggtree(as.phylo(hcTree), ladderize = FALSE, layout = 'dendrogram')
  )
  
  return(list(tree = hcTree, order = hclustObj$order))
}


#' Generate Hierarchical Groups from Clustering Tree
#' 
#' Creates variable groups for overlapping group lasso based on hierarchical
#' clustering tree structure, optionally using proportions or means.
#' @param tree A hierarchical clustering tree generated from `generateTree`.
#' @param nclust Integer; number of clusters to generate.
#' @param proportions Data frame of cell type proportions.
#' @param means Data frame of marker means.
#' @param type Character; group type to generate ("all", "prop", or "mean").
#' @return List of variable groups for overlapping group lasso.
#' @importFrom janitor make_clean_names
#' @noRd
generateGroups <- function(tree, nClust = 20, proportions, means, type = "all") {
  
  # Extract clusters and assign group numbers
  subGroups <- tree$data$clusters
  data <- tree$data
  data$group <- seq_along(subGroups)
  
  # Initialize list for storing variable groups
  varGroups <- vector("list", length(subGroups))
  
  # Create groups based on cell type proportions
  for (i in seq_along(subGroups)) {
    varGroups[[i]] <- which(
      janitor::make_clean_names(colnames(proportions)) %in% 
        janitor::make_clean_names(subGroups[[i]])
    )
  }
  
  if (type == "prop") return(varGroups)
  
  # Create additional groups based on mean marker expressions
  lastMax <- nClust
  offset <- length(varGroups)
  
  for (i in seq_len(ncol(means))) {
    index <- offset + i
    begin <- lastMax + 1
    
    # Add a new group containing a sequence from `begin` to `begin`
    varGroups[[index]] <- seq(begin, begin)
    
    # Update lastMax for the next iteration
    lastMax <- begin
  }
  
  return(varGroups)
}

#' Get Significant Features from Group Lasso Model
#' 
#' Extracts significant features from a fitted group lasso model based on either
#' mean values or proportions, performs statistical tests, and adjusts p-values.
#' @param fit Fitted group lasso model.
#' @param type Character; specifies whether to use "mean" values or "prop" (proportions), default is "mean".
#' @param mean Data frame of mean values for each feature, default is NULL.
#' @param prop Data frame of proportion values for each feature, default is NULL.
#' @param clinicalData Data frame of clinical data to join with significant features, default is NULL.
#' @param outcome Character; outcome variable used for group comparisons.
#' @return List containing:
#'   \item{stats}{Data frame of statistical test results for each feature.}
#'   \item{sigFeatures}{Data frame of significant features with expression/proportion values.}
#' @importFrom dplyr mutate select filter left_join group_by
#' @importFrom ggpubr t_test adjust_pvalue add_significance add_xy_position
#' @importFrom reshape2 melt
#' @noRd
getSigFeatures <- function(fit, type = "mean", mean = NULL, clinicalVariables = NULL,
                           prop = NULL, clinicalData = NULL, outcome = NULL) {
  
  # Get coefficients from the model and filter out zero values
  coefsSub <- coef(fit) %>%
    as.matrix() %>%
    as.data.frame() %>%
    dplyr::mutate(feature = rownames(.)) %>%
    dplyr::filter(V1 != 0)
  
  if (type == "mean") {
    
    coefsSub <- coefsSub %>%
      dplyr::filter(!feature %like% '_logit') %>%
      dplyr::filter(!feature %like% 'Intercept')  %>%
      dplyr::filter(!feature %in% clinicalVariables)
    
    sigFeatures <- mean %>%
      dplyr::select(coefsSub$feature) %>%
      mutate(sample_id = as.integer(rownames(.))) %>%
      left_join(clinicalData)
    
    stats <- sigFeatures %>% 
      group_by(!!dplyr::sym(outcome)) %>%
      dplyr::select(coefsSub$feature) %>%
      as.data.frame() %>%
      melt() %>%
      as.data.frame() %>%
      mutate(feature = variable) %>%
      mutate(outcome = !!dplyr::sym(outcome)) %>%
      group_by(feature) %>%
      t_test(value ~ outcome) %>%
      adjust_pvalue(method = "fdr") %>%
      add_significance("p.adj") %>%
      add_xy_position(x = 1, dodge = 0.0) %>%
      dplyr::mutate(p.adj = round(p.adj, 2))
    
  } else if (type == "prop") {
    
    coefsSub <- coefsSub %>%
      dplyr::filter(feature %like% '_logit') %>%
      dplyr::filter(!feature %like% 'Intercept') 
    
    sigFeatures <- prop %>%
      dplyr::select(coefsSub$feature) %>%
      mutate(sample_id = as.integer(rownames(.))) %>%
      left_join(clinicalData)
    
    stats <- sigFeatures %>% 
      group_by(!!dplyr::sym(outcome)) %>%
      dplyr::select(coefsSub$feature) %>%
      as.data.frame() %>%
      melt() %>%
      as.data.frame() %>%
      mutate(feature = variable) %>%
      mutate(outcome = !!dplyr::sym(outcome)) %>%
      group_by(feature) %>%
      t_test(value ~ outcome) %>%
      adjust_pvalue(method = "fdr") %>%
      add_significance("p.adj") %>%
      add_xy_position(x = 1, dodge = 0.0) %>%
      dplyr::mutate(p.adj = round(p.adj, 2))
  }
  
  colnames(sigFeatures) <- str_replace_all(colnames(sigFeatures), "_", " ")
  
  stats$feature <- str_replace_all(stats$feature, "_", " ")
  
  return(list(stats = stats, sigFeatures = sigFeatures))
}

#' Compute Features for Group Lasso Model
#' 
#' Generates feature matrices for group lasso based on either cell type proportions
#' or mean marker expression. Proportions are optionally logit-transformed.
#' @param sce SingleCellExperiment object containing cell data.
#' @param featureType Character; feature type to compute, either "prop" or "mean".
#' @param cellTypeCol Character; column name for cell type labels.
#' @param sampleCol Character; column name for sample identifiers.
#' @param logit Logical; if TRUE, applies logit transformation to proportions.
#' @param useMarkers Character vector of marker names.
#' @param assay Character; assay name from which to extract marker values.
#' @return Data frame of features for group lasso model.
#' @importFrom SummarizedExperiment colData assay
#' @importFrom dplyr group_by summarise_at left_join mutate
#' @importFrom tidyr pivot_longer pivot_wider
#' @importFrom tibble column_to_rownames
#' @noRd
computeFeatures <- function(sce, featureType = "prop", cellTypeCol = "clusters",
                            sampleCol = "sample_id", logit = TRUE, useMarkers, assay = "norm") {
  if (featureType == "prop") {
    features <- getProp(sce, feature = cellTypeCol, imageID = sampleCol, logit = logit)
    colnames(features) <- paste0(colnames(features), "_logit")
    
  } else if (featureType == "mean") {
    # Extract cell data and select marker columns
    colData <- as.data.frame(SummarizedExperiment::colData(sce))
    markerData <- colData %>%
      dplyr::select(sample_id = !!dplyr::sym(sampleCol), !!dplyr::sym(cellTypeCol)) %>%
      cbind(data.frame(t(SummarizedExperiment::assay(sce, assay)), check.names = FALSE)) %>%
      dplyr::select(sample_id, !!dplyr::sym(cellTypeCol), useMarkers)
    
    # Compute mean marker expression per cell type
    features <- markerData %>%
      dplyr::group_by(sample_id, !!dplyr::sym(cellTypeCol)) %>%
      dplyr::summarise_at(vars(-group_cols()), mean, na.rm = TRUE) %>%
      tidyr::pivot_longer(-c(sample_id, !!dplyr::sym(cellTypeCol)), names_to = "markers") %>%
      tidyr::pivot_wider(names_from = c(!!dplyr::sym(cellTypeCol), markers), values_from = value) %>%
      tibble::column_to_rownames("sample_id") %>%
      dplyr::mutate(across(everything(), ~replace_na(., 0)))
  }
  
  return(features)
}

