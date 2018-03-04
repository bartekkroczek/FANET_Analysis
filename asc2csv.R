library('eyelinker')  
setwd("~/Dropbox/Data/FAN_ET/Badanie P/2017-05-06_Badanie_P/BadanieP_FAN_ET/Dane trackingowe/")

file.names <- dir(".", pattern =".asc")
print(file.names)

file.done <- c("101FEMALE19.asc", "100MALE24.asc", "104FEMALE23.asc", "106MALE25.asc", "107FEMALE21.asc", "108FEMALE20.asc",
                "10MALE26.asc", "110FEMALE20.asc", "112FEMALE22.asc", "113MALE26.asc", "117MALE20.asc", "119FEMALE19.asc",
                "120FEMALE29.asc", "122FEMALE23.asc", '123FEMALE18.asc', '124MALE22.asc', "129FEMALE19.asc", "131FEMALE20.asc",
                "133FEMALE35.asc", "134FEMALE18.asc", "135MALE20.asc", "139MALE20.asc", "145FEMALE26.asc", "146FEMALE22.asc",
                "148FEMALE20.asc", "154MALE20.asc", "155MALE33.asc", "157MALE21.asc", "15FEMALE25.asc", "162FEMALE18.asc", 
                "163MALE40.asc", "164FEMALE21.asc", "171FEMALE22.asc", "173MALE28.asc", "174FEMALE21.asc", "178FEMALE23.asc",
                "17MALE19.asc", "180FEMALE32.asc", "18FEMALE32.asc", "24FEMALE28.asc", "27MALE22.asc", "30FEMALE23.asc",
                "36FEMALE20.asc", "38FEMALE20.asc", "40FEMALE21.asc", "44MALE24.asc", "46FEMALE19.asc", "51FEMALE33.asc", 
                "55FEMALE24.asc", "56MALE19.asc", "76MALE18.asc", "77FEMALE26.asc", "79MALE22.asc", "82FEMALE21.asc", 
                "85FEMALE18.asc", "89MALE23.asc", "90FEMALE18.asc", "92FEMALE19.asc", "94MALE30.asc", "98FEMALE23.asc",
                "136FEMALE23.asc")

file.toconv <- mapply('-', file.names, file.done, SIMPLIFY = FALSE)
file.toconv <- c('12MALE21.asc', '14FEMALE19.asc', '62FEMALE39.asc', '83MALE27.asc', '130MALE18.asc')

print(file.toconv)

#for(i in 1:length(file.names)){
#  print(gsub(".asc", "", file.names[i]))
#  f <- read.asc(file.names[i])
#  write.csv(f$raw, file = gsub(".asc", "_raw.csv", file.names[i]))
#  write.csv(f$sacc, file = gsub(".asc", "_sacc.csv", file.names[i]))
#  write.csv(f$fix, file = gsub(".asc", "_fix.csv", file.names[i]))
#} 

lapply(file.toconv, convert)
convert <- function(filename) {
  out <- tryCatch(
    {
      
      message(paste("Processed at:", gsub(".asc", "", filename)))
      f <- read.asc(filename)
      write.csv(f$raw, file = gsub(".asc", "_raw.csv", filename))
      write.csv(f$sacc, file = gsub(".asc", "_sacc.csv", filename))
      write.csv(f$fix, file = gsub(".asc", "_fix.csv", filename))
      rm("f")
      gc()
    },
    error=function(cond) {
      message(paste("Problem with:", filename))
      message("Here's the original error message:")
      message(cond)
      # Choose a return value in case of error
      return(NA)
    },
    warning=function(cond) {
      message(paste("File caused a warning:", filename))
      message("Here's the original warning message:")
      message(cond)
      # Choose a return value in case of warning
      return(NULL)
    },
    finally={
      message(paste("Processed File:", filename))
    }
  )    
  return(out)
}
