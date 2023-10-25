from sparknlp.pretrained import PretrainedPipeline

if __name__ == '__main__':
    pipeline = PretrainedPipeline("recognize_entities_dl_noncontrib", 'en')
    result = pipeline.annotate('Google has announced the release of a beta version of the popular TensorFlow machine learning library.')
    print(result['ner'])
    print(result['entities'])
