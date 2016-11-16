split("\n") 
    | .[1:]
    | map(split(","))
    | map([
        (.[0] | tonumber), # Move the `test_result` header to the last position
        (.[1] | tonumber),
        (.[3] | tonumber),
        (.[4] | tonumber),
        (.[5] | tonumber),
        (.[6] | tonumber),
        (.[7] | tonumber),
        (.[8] | tonumber),
        .[2] 
    ])
    | [("plasma_glucose,bp,age,skin_thickness,num_pregnancies,insulin,bmi,pedigree,test_result" | split(",")), .[]] # CSV Header
    | map(@csv)
    | join("\n")

    # Map it to a JSON object
    # | map({
    #       "plasma_glucose": .[0] | tonumber,
    #       "bp": .[1] | tonumber,
    #       "test_result": .[2],
    #       "skin_thickness": .[3] | tonumber,
    #       "num_pregnancies": .[4] | tonumber,
    #       "insulin": .[5] | tonumber,
    #       "bmi": .[6] | tonumber,
    #       "pedigree": .[7] | tonumber,
    #       "age": .[8] | tonumber
    #   })