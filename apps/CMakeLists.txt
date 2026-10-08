add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
add_executable(ninfer-asr asr/main.cpp)
add_executable(ninfer-align alignment/main.cpp)
ninfer_internal_includes(ninfer-align)
target_link_libraries(ninfer-align PRIVATE ninfer_engine ninfer::json)
add_executable(ninfer-embed embedding/main.cpp)
ninfer_internal_includes(ninfer-embed)
target_link_libraries(ninfer-embed PRIVATE ninfer_engine ninfer::json)
ninfer_internal_includes(ninfer-asr)
target_link_libraries(ninfer-asr PRIVATE ninfer_engine ninfer::json)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)
