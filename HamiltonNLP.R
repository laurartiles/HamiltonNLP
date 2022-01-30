# If packages are not installed, please proceed and install them ###############3
install.packages("rJava")
install.packages("quanteda")
install.packages("tokenizers.bpe")
install.packages("RColorBrewer")
install.packages("wordcloud2")
install.packages("kableExtra")
install.packages("gutenbergr")
install.packages("nametagger")
install.packages("qdap")
install.packages("ggmap")
install.packages("rworldmap")
install.packages("NLP")
install.packages("tm")
install.packages("udpipe")

library(utf8)
library(tm)
library(ggplot2)
library(wordcloud)
library(reshape2)
library(quanteda)
library(udpipe)
library("spacyr")
library(reticulate)
library("tokenizers.bpe")
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
library(dplyr)
library(kableExtra)
library(gutenbergr)
library(nametagger)
library(qdap)
library(rvest)
library(NLP)
library(openNLP)
library(ggmap)
library(rworldmap)
library(rJava)
library(stringr)

############################## The corpus #########################
# We create our data set form the file online
# The lines are created and encoded in UTF-8
hamilton <- "https://raw.githubusercontent.com/amandavisconti/ham4corpus/master/All_Lyrics_Speakers"
lines <- readLines(hamilton,
                   encoding = "UTF-8") #It takes a few seconds

# We show the first 20 lines
# The name of the character is shown above their part
lines[1:20]

# How many lines are there? 
length(lines) #5409

# Show the lines in one paragraph
paste(lines[1:20], collapse = " ")

################################ Check encoding ##############

# Make sure all lines are correctly encoded in UTF-8
lines[!utf8_valid(lines)] #character(0) ==> All lines are made of correct UTF-8 characters

#Check character normalization. Specifically, the normalized composed form (NFC)
lines_NFC <- utf8_normalize(lines)
sum(lines_NFC != lines) #0 means all right. The text is in NFC.

################################ Cleaning data ################

text_clean <- lines

#Remove possible special symbols
text_clean <- gsub("_", "", text_clean)
text_clean <- gsub("-", "", text_clean)

# Transform to Lowercase & remove punctuation 
text_clean <- tolower(text_clean) 
text_clean <- removePunctuation(text_clean)

# List standard English stop words
stopwords("en")

# Add new words to the list: new_stops
# These contractions appear when just the standard stopwords are removed
# This is why I add them to my list of stopwords
new_stops <- c("'m", "'re", "'s", "'ve","'d", "'ll","gon'", stopwords("en"))
new_stops

# Remove stop words from text
text_clean <- removeWords(text_clean, new_stops)

# Remove excess whitespace
text_clean <-stripWhitespace(text_clean)

# Compare the text with and without cleaning
head(text_clean)
head(lines)

# Obtain a vector with all the lines in the file
stringHamilton <- paste(text_clean, collapse = " ") 
head(stringHamilton)


######################## Number of occurrences #####################
# Number of times that each character sings
result= sum(str_count(lines, "BURR"))
result

result2= sum(str_count(lines, "ANGELICA"))
result2

result3= sum(str_count(lines, "HAMILTON"))
result3

result4= sum(str_count(lines, "ELIZA"))
result4

################################ Tokenize ###################
## Install spacyr if not already
spacy_install()
spacy_initialize()

#spacy_uninstall()
#spacy_finalize()

tokens <- spacy_tokenize(lines
                         #Parameters asigned by default:
                         #remove_punct = FALSE, punt symbols are tokens
                         #remove_url = FALSE, url elements are tokens
                         #remove_numbers = FALSE, numbers are tokens
                         #remove_separators = TRUE, spaces are NOT tokens
                         #remove_symbols = FALSE, symbols (like €) are tokens
)

#Returns a list
v_tokens <- unlist(tokens)
v_tokens[1:20]

# NUMBER OF TOKENS (WITH REPETITION AND WITHOUT REPETITION)
length(v_tokens) # 13626 tokens (many repeated)
length(unique(v_tokens)) #2842 different (unique) tokens.

# We show the 25 first tokes with more ocurrencies
head(sort(table(v_tokens), decreasing = TRUE), n = 25)

#Create a simple plot
plot(head(sort(table(v_tokens), decreasing = TRUE), n = 20),
     xlab = "Token",
     ylab = "Ocurrences"
)

########################## POS -> spacyr ######################

# Not cleaned text
string <- paste(lines, collapse = " ") 
head(string)

sum(duplicated(names(string)))
names(string) <- NULL

begin <- Sys.time()

#returns a list of dataframes
Sys.time()-begin
tic <- Sys.time()

res1 <- lapply(string,
              spacy_parse, #This is the function to apply to every element in string
              dependency = TRUE, nounphrase = TRUE )

df <- res1[[1]] #A data frame with the first results

for (i in 1:length(res1)){ #Attention! The loop starts from 2
  df <- rbind(df, res1[[i]]) 
  }

Sys.time()-tic

#As this takes a while, I save the result
saveRDS(df, file="spacy_parse_Hamilton.rds")

#Shows the first 20 tokens.
#The first 2 cols UNSHOWN are doc_id and sentence_id, we start in column 3
kable_styling(kable(df[1:20, c(3:ncol(df))]), font_size = 12)

########################### POS -> udpipe  #########################

model_en<-udpipe_download_model(language= "english")
udmodel_en<-udpipe_load_model(file = model_en$file_model)

tic <- Sys.time()
pos_1<-udpipe_annotate(udmodel_en, x = string, parallel.cores = 10) #Check your system!!
df<-as.data.frame(pos_1)
Sys.time()-tic

#Write the result as a coNLL file
cat(pos_1$conllu, file = "udpipes_en_hamilton.conllu")

#Show the annotations of the first 20 tokens
#df has 14 columns, we show them in two tables
#The first 4 cols UNSHOWN are doc_id, paragraph_id, sentence_id and sentence
kable_styling(kable(df[1:20, c(5:9)]), font_size = 15 )

#Remaining cols
kable_styling(kable(df[1:20, c(10:14)]), font_size = 15)              

########################### Wordclouds #######################

# Create Corpus from lines 
myCorpus<-Corpus(VectorSource(lines)) #your file into a corpus

# Clean the text data with a function: remove numbers, Punctuation, whitespaces
# Convert all in lower cases and remove stopwords
clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, new_stops)
  return(corpus)
}

# Apply your customized function to the corpus
clean_corpus <- clean_corpus(myCorpus)

# Create TermDocumentMatrix
dtm <- TermDocumentMatrix(clean_corpus) 
inspect(dtm[1:9, 1:9])
matrix <- as.matrix(dtm)
words <- sort(rowSums(matrix),decreasing=TRUE) 
df2 <- data.frame(word = names(words),freq=words)
df2
# Create WordCloud 
wordcloud2(data=df2, size=1.0, color='random-dark')


##############################  BPE model  ##################

# We divide the text in half to use the first half to train the BPE model
# and the second half to test the model 

final = length(text_clean)
div = round(length(text_clean)/2)

text_clean[1:div]
text_part1 <- paste(unlist(text_clean[1:div]), collapse=" ")
text_part1 <-stripWhitespace(text_part1)
head(text_part1)

text_clean[(div+1):final]
text_part2 <- paste(unlist(lines[(div+1):final]), collapse=" ")
text_part2 <-stripWhitespace(text_part2)
head(text_part2)

#I can't use a single line with all the text (text_part1), but I can use a vector of chapters
model <- bpe(unlist(text_clean[1:div]))

#We apply the model to the second part of the text (here we can use a single string)
subtoks2 <- bpe_encode(model, x = text_part2, type = "subwords")
head(unlist(subtoks2), n=50)

sessionInfo()
