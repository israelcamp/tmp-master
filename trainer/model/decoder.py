import torch


class DecoderBase(torch.nn.Module):

    def __init__(self,):
        super().__init__()

    def forward(self, encoder_last_hidden_state, labels=None, decoder_input_ids=None):
        encoder_outputs = (encoder_last_hidden_state,)
        outputs = self.decoder(encoder_outputs=encoder_outputs, labels=labels, decoder_input_ids=decoder_input_ids)
        return outputs

    @torch.no_grad()
    def generate(self, encoder_last_hidden_state, max_length=32, start_token_id=0, eos_token_id=1):
        self.eval()

        decoded_ids = torch.full((encoder_last_hidden_state.shape[0], 1),
                                 start_token_id,
                                 dtype=torch.long).to(encoder_last_hidden_state.device)

        for step in range(max_length):
            logits = self(decoder_input_ids=decoded_ids,
                            encoder_last_hidden_state=encoder_last_hidden_state)[0]
            next_token_logits = logits[:, -1, :]

            # Greedy decoding
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)

            # Check if output is end of senquence for all batches
            if torch.eq(next_token_id[:, -1], eos_token_id).all():
                break

            # Concatenate past ids with new id, keeping batch dimension
            decoded_ids = torch.cat([decoded_ids, next_token_id], dim=-1)

        return decoded_ids
