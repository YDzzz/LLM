## Document Loader Interface
| Method Name | Explanation                                                                                                                 |
|-------------|-----------------------------------------------------------------------------------------------------------------------------|
| lazy_load	  | Used to load documents one by one lazily. Use for production code.                                                          |
| alazy_load  | Async variant of lazy_load                                                                                                  |
| load        | Used to load all the documents into memory eagerly. Use for prototyping or interactive work.                                |
| aload	      | Used to load all the documents into memory eagerly. Use for prototyping or interactive work. Added in 2024-04 to LangChain. |
- The load methods is a convenience method meant solely for prototyping work – it just invokes list(self.lazy_load()).
- The alazy_load has a default implementation that will delegate to lazy_load. If you’re using async, we recommend overriding the default implementation and providing a native async implementation.

