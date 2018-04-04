library(readr)
library(ggplot2)
library(scales)
library(dplyr)
library(tidyr)

FOLDER <- "Linux2-cut_0.1"
FEATURE <- 'n_gram'
LAYER <- 1



#########################
# PLOTTING ACTIVATIONS  #
#########################
# activations <-read_csv(paste0(FOLDER, "/activations.csv"), col_names = TRUE)
#
# # Clustering
# activation_mat <- activations %>%
#                   filter(grepl('^_', Series)) %>%
#                   spread(Series, Value)
# activation_mat$time <- NULL
# head(activation_mat)
# activation_mat <- as.matrix(activation_mat)
#
# heatmap(activation_mat[0:50,], Rowv=NA)

#########################
# PLOTTING ATTRIBUTIONS #
#########################
attributions <-read_csv(paste0(FOLDER, "/attributions"), col_names = TRUE)

# Gets scores
attributions[['Neuron']] <- factor(0:(nrow(attributions)-1),
                           levels = 0:(nrow(attributions)-1),
                           labels = as.character(0:(nrow(attributions)-1)))

data <- gather(attributions, key='Feature', value='Score', -Neuron)


# Gets selected ones
data$selected <- ifelse(data$Score > .1, 'X', '')

# Filters
data <- data %>% filter(grepl(FEATURE,Feature)) %>%
                mutate(layer=as.numeric(Neuron) %/% 128) %>% filter(layer==LAYER)
print(head(data))

# Plots Activations
p <- ggplot(data, aes(x=Feature, y=Neuron, fill=Score, label=selected)) +
  geom_bin2d(aes='identity') +
  geom_text(color='red') +
  scale_fill_gradient2(midpoint=median(data$Score),
                       low = muted("blue"), mid = "white",high = muted("red"),
                       limits=c(0,NA)) +
  facet_grid(layer~., scales = 'free') +
  theme(axis.text.x = element_text(angle=90))

print(p)

# # Tries to cluster the neurons
# attr_to_cluster <- attributions
# rownames(attr_to_cluster) <- attr_to_cluster[['Neuron']]
# attr_to_cluster[['Neuron']] <- NULL
# attr_to_cluster <-as.matrix(attr_to_cluster)
# print(attr_to_cluster)
# heatmap(attr_to_cluster, Colv=NA)

############################################
# Selective plotting of neuron activations #
############################################
activations <-read_csv(paste0(FOLDER, "/activations.csv"), col_names = TRUE)

# Clustering
to_plot <- activations %>%
                  filter(grepl('^_182', Series) | grepl('n_gram_0', Series) ) %>%
                  filter(time < 800 & time > 400)

p <- ggplot(to_plot, aes(x=time, y=Value, fill=Series, color=Series, shape = Series)) +
            scale_y_continuous(limits = c(-1,1), breaks=c(-1,1)) +
            geom_line() +
            geom_point()# +
            #facet_grid(Series~., scales=\"free\")
p