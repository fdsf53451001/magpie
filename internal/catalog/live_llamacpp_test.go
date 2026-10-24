package catalog

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"testing"
)

// llama.cpp's llama-server (build 11840, started with -c 131072 and
// --mmproj) lists its model with the window it serves in meta.n_ctx and
// its image input in architecture.input_modalities, not in context_length
// or modalities: the list keeps both.
const llamaServerModels = `{"models":[{"name":"gemma-4-qat","model":"gemma-4-qat","modified_at":"","size":"","digest":"","type":"model","description":"","tags":[""],"capabilities":["completion","multimodal"],"parameters":"","details":{"parent_model":"","format":"gguf","family":"","families":[""],"parameter_size":"","quantization_level":""}}],"object":"list","data":[{"id":"gemma-4-qat","aliases":["gemma-4-qat"],"tags":[],"object":"model","architecture":{"input_modalities":["text","image","video"],"output_modalities":["text"]},"created":1792780414,"owned_by":"llamacpp","meta":{"vocab_type":2,"n_vocab":262144,"n_ctx":131072,"n_ctx_train":262144,"n_embd":2816,"n_params":25233142046,"size":14233222264,"ftype":"Q4_0"}}]}`

// A llama-server router (--models-dir) lists a model it hasn't loaded with
// no meta and text alone; a loaded text model with its window; an odd
// meta and architecture don't lose the list. Another vendor's row with the
// same fields is left as it was.
const llamaRouterModels = `{"object":"list","data":[
  {"id":"qwen3.6-35b-a3b","aliases":[],"tags":[],"object":"model","owned_by":"llamacpp","created":1792780414,"status":{"value":"unloaded","args":[]},"architecture":{"input_modalities":["text"],"output_modalities":["text"]},"source":"cache","can_remove":true},
  {"id":"glm-5.2-air","aliases":[],"tags":[],"object":"model","owned_by":"llamacpp","created":1792780414,"status":{"value":"loaded","args":[]},"architecture":{"input_modalities":["text"],"output_modalities":["text"]},"meta":{"n_ctx":65536,"n_ctx_train":131072}},
  {"id":"odd-llama","object":"model","owned_by":"llamacpp","meta":{"n_ctx":"lots"},"architecture":"text"},
  {"id":"zero-llama","object":"model","owned_by":"llamacpp","meta":{"n_ctx":0}},
  {"id":"relay-model","object":"model","owned_by":"relay","meta":{"n_ctx":8192},"architecture":{"input_modalities":["text","image"]}}]}`

func TestFetchLlamaServerWindowAndImages(t *testing.T) {
	for name, c := range map[string]struct {
		list string
		want map[string]string
	}{
		"single model": {llamaServerModels, map[string]string{"gemma-4-qat": "131072 true"}},
		"router": {llamaRouterModels, map[string]string{
			"qwen3.6-35b-a3b": "0 <nil>",
			"glm-5.2-air":     "65536 <nil>",
			"odd-llama":       "0 <nil>",
			"zero-llama":      "0 <nil>",
			"relay-model":     "0 <nil>",
		}},
	} {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
				rw.Write([]byte(c.list))
			}))
			defer srv.Close()
			ms, _, err := FetchAt(context.Background(), srv.URL+"/v1", "k", false, nil)
			if err != nil {
				t.Fatal(err)
			}
			got := map[string]string{}
			for _, m := range ms {
				images := "<nil>"
				if m.ImageInput != nil {
					images = fmt.Sprint(*m.ImageInput)
				}
				got[m.ID] = fmt.Sprint(m.Context, " ", images)
			}
			if !maps.Equal(got, c.want) {
				t.Errorf("got %v\nwant %v", got, c.want)
			}
		})
	}
}
