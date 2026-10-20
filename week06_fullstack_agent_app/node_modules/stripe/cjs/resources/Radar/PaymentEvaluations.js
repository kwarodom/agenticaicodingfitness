"use strict";
// File generated from our OpenAPI spec
Object.defineProperty(exports, "__esModule", { value: true });
exports.PaymentEvaluations = void 0;
const StripeResource_js_1 = require("../../StripeResource.js");
const stripeMethod = StripeResource_js_1.StripeResource.method;
exports.PaymentEvaluations = StripeResource_js_1.StripeResource.extend({
    create: stripeMethod({
        method: 'POST',
        fullPath: '/v1/radar/payment_evaluations',
    }),
});
