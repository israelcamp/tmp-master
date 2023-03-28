from ctc import GreedyCTCDecoder

def stop_on_eos(input_ids):
    fixed = []
    for iids in input_ids:
        try:
            idx = iids.index(1)
        except:
            fixed.append(iids)
        else:
            fixed.append(iids[:idx+1])
    return fixed

def compute_ctc(batch, model, tokenizer, ctc_decoder):
    
    images, attention_image = batch
    
    embeddings = model(images, attention_image)
    decoded_ids = embeddings.argmax(-1)#.squeeze()

    # attention_image = attention_image or len(decoded_ids) * [None]
    decoded = [
        ctc_decoder(dec, att) for dec, att in zip(decoded_ids, attention_image)
    ]

    decoded = stop_on_eos(decoded)
    preds = tokenizer.batch_decode(decoded, skip_special_tokens=True)
    return preds